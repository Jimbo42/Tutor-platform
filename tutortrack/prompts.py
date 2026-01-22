import re
import sqlite3
from openai import Client
import streamlit as st
from streamlit import session_state as ss
from fpdf import FPDF
import json
from datetime import datetime
from pathlib import Path

from tutortrack.lessons import get_conn
from shared.published_db import publish_item

# Tutor root
ROOT = Path(__file__).resolve().parents[1]

# Shared PDF output folder
PDF_DIR = ROOT / "shared" / "pdf_files"
PDF_DIR.mkdir(parents=True, exist_ok=True)

# Font path
FONT_PATH = ROOT / "tutortrack" / "resources" / "fonts" / "DejaVuSans.ttf"

DEFAULT_LLM_CONFIG = {
    "openai_model": "gpt-4o-mini",
    "temperature": 0.0,
    "top_p": 1.0,
    "presence": 0.0,
    "frequency": 0.0,
}

MODEL_PRESETS = {
    "Precise (Math / Science)": {
        "openai_model": "gpt-4o-mini",
        "temperature": 0.0,
        "top_p": 1.0,
        "presence": 0.0,
        "frequency": 0.0,
    },
    "Balanced": {
        "openai_model": "gpt-4o-mini",
        "temperature": 0.3,
        "top_p": 0.95,
        "presence": 0.1,
        "frequency": 0.1,
    },
    "Creative": {
        "openai_model": "gpt-4o",
        "temperature": 0.7,
        "top_p": 0.9,
        "presence": 0.4,
        "frequency": 0.3,
    },
}

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "AIDA.db"

def db_conn():
    return sqlite3.connect(DB_PATH)

def list_templates():
    conn = db_conn()
    cur = conn.cursor()
    rows = cur.execute("SELECT id, title, category, model, updated_at FROM ChatTemplates ORDER BY title ASC").fetchall()
    conn.close()
    return rows

def load_template(template_id: int):
    conn = db_conn()
    cur = conn.cursor()
    row = cur.execute("""
        SELECT id, title, category, model, system_prompt, user_prompt, fields_json, params_json, updated_at
        FROM ChatTemplates WHERE id=?
    """, (template_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "title": row[1],
        "category": row[2],
        "model": row[3],
        "system_prompt": row[4],
        "user_prompt": row[5],
        "fields": json.loads(row[6] or "[]"),
        "params": json.loads(row[7] or "{}"),
        "updated_at": row[8],
    }

def upsert_template_from_json(tpl: dict):
    required = ["title", "model", "system_prompt", "user_prompt", "fields", "params"]
    for k in required:
        if k not in tpl:
            raise ValueError(f"Template missing required key: {k}")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ChatTemplates (title, category, model, system_prompt, user_prompt, fields_json, params_json, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(title) DO UPDATE SET
            category=excluded.category,
            model=excluded.model,
            system_prompt=excluded.system_prompt,
            user_prompt=excluded.user_prompt,
            fields_json=excluded.fields_json,
            params_json=excluded.params_json,
            updated_at=excluded.updated_at
    """, (
        tpl["title"].strip(),
        tpl.get("category"),
        tpl["model"].strip(),
        tpl["system_prompt"],
        tpl["user_prompt"],
        json.dumps(tpl["fields"], ensure_ascii=False),
        json.dumps(tpl["params"], ensure_ascii=False),
        now
    ))
    conn.commit()
    conn.close()

def export_template_to_json(template_id: int) -> str:
    tpl = load_template(template_id)
    if not tpl:
        raise ValueError("Template not found")
    # Remove internal DB keys for portability
    portable = {
        "title": tpl["title"],
        "category": tpl["category"],
        "model": tpl["model"],
        "system_prompt": tpl["system_prompt"],
        "user_prompt": tpl["user_prompt"],
        "fields": tpl["fields"],
        "params": tpl["params"],
    }
    return json.dumps(portable, ensure_ascii=False, indent=2)

def get_template_list():
    conn_l = get_conn()
    cl = conn_l.cursor()
    with conn_l:
        cl.execute("SELECT * FROM ChatConfig ORDER BY Title ASC")
        data = cl.fetchall()
        ss.template_list = data

    conn_l.close()

def fill_prompt_with_highlight(prompt: str, values: dict, fields: list):
    """
    Fill a prompt, but render missing fields as highlighted <Field Label> spans.
    Returns HTML.
    """
    out = prompt

    # Build map name -> label
    name_to_label = {}
    for f in fields:
        name_to_label[f["name"]] = f.get("label", f["name"])

    for name, label in name_to_label.items():
        val = values.get(name)

        if val in (None, "", []):
            replacement = f'<span class="tpl-missing">&lt;{label}&gt;</span>'
        else:
            replacement = f'<span class="tpl-filled">{val}</span>'

        out = re.sub(r"{{\s*" + re.escape(name) + r"\s*}}", replacement, out)

    # Escape any leftover angle brackets that are not ours
    return out

def edit_user_template(mode):
    if mode == "t":
        new_text = ss.user_template.replace( ss.template_list[ss.index][8], ss.p_text)
        ss.user_template = new_text
    if mode == "n":
        new_text = ss.user_template.replace( ss.template_list[ss.index][5], str(ss.p_number))
        ss.user_template = new_text
    if mode == "p":
        new_text = ss.user_template.replace( ss.template_list[ss.index][6], ss.p_paragraph)
        ss.user_template = new_text

    # was this the last field in the template ?
    if re.search(r"<.*?>", ss.user_template) is None:
        ss.prompt = ss.user_template

def reset_template():
    ss.user_template = None

def reset_prompt():
    ss.prompt = None
    ss.user_template = None
    ss.template = None
    ss.show_template = False

_MATH_BLOCK_OR_INLINE = re.compile(r"(\$\$.*?\$\$|\$.*?\$)", re.DOTALL)

# Heuristic: a contiguous "TeX-ish" run starting with a command like \text, \frac, \sqrt, etc.
_TEX_RUN = re.compile(
    r"(?:\\[A-Za-z]+(?:\{[^}]*\})*[A-Za-z0-9{}\[\]_^%:+\-*/().=,]*)+"
)

def translate_latex(val: str) -> str:
    if "\\" not in val and "$" not in val:
        return val

    s = val

    # 1) Normalize \( \) and \[ \] into $ and $$ (Streamlit-friendly)
    s = re.sub(r"\\\((.*?)\\\)", r"$\1$", s, flags=re.DOTALL)
    s = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", s, flags=re.DOTALL)

    # 2) Wrap bare TeX commands (e.g., \text{C}_2\text{H}_2\text{O}_4) in $...$
    #    but ONLY outside existing $...$ / $$...$$ regions.
    parts = _MATH_BLOCK_OR_INLINE.split(s)

    def wrap_tex_runs(text: str) -> str:
        def repl(m: re.Match) -> str:
            expr = m.group(0)

            # keep trailing punctuation outside the $...$
            m2 = re.match(r"^(.*?)([.,;:!?)]*)$", expr)
            core, punct = m2.group(1), m2.group(2)

            return f"${core}$" + punct

        return _TEX_RUN.sub(repl, text)

    for i in range(len(parts)):
        # Even indices are outside math (because split keeps the delimiters)
        if i % 2 == 0:
            parts[i] = wrap_tex_runs(parts[i])

    return "".join(parts)

JINJA_VAR = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")

def render_field(field, key_prefix="tpl_"):
    name = field["name"]
    label = field.get("label", name)
    ftype = field.get("type", "text")
    default = field.get("default", "")

    k = f"{key_prefix}{name}"

    # If Streamlit already has state for this widget, DO NOT pass value=
    has_state = k in ss

    if ftype == "text":
        if has_state:
            return st.text_input(label, key=k)
        else:
            return st.text_input(label, value=str(default), key=k)

    if ftype == "textarea":
        if has_state:
            return st.text_area(label, key=k, height=140)
        else:
            return st.text_area(label, value=str(default), key=k, height=140)

    if ftype == "number":
        try:
            d = int(default) if default != "" else 0
        except Exception:
            d = 0

        if has_state:
            return st.number_input(label, step=1, key=k)
        else:
            return st.number_input(label, value=d, step=1, key=k)

    if ftype == "select":
        options = field.get("options", [])
        if not options:
            options = [str(default)] if default else []

        if has_state:
            return st.selectbox(label, options, key=k)
        else:
            idx = 0
            if default in options:
                idx = options.index(default)
            return st.selectbox(label, options, index=idx if options else None, key=k)

    if ftype == "checkbox":
        d = bool(default)
        if has_state:
            return st.checkbox(label, key=k)
        else:
            return st.checkbox(label, value=d, key=k)

    # fallback
    if has_state:
        return st.text_input(label, key=k)
    else:
        return st.text_input(label, value=str(default), key=k)

def fill_prompt(prompt: str, values: dict) -> str:
    out = prompt
    for k, v in values.items():
        out = re.sub(r"{{\s*" + re.escape(k) + r"\s*}}", str(v), out)
    return out

def missing_vars(prompt: str, values: dict):
    needed = set(JINJA_VAR.findall(prompt or ""))
    missing = [v for v in sorted(needed) if v not in values or values[v] in (None, "", [])]
    return missing

def validate_questions_schema(obj: dict):
    """
    Validates the worksheet JSON schema.
    Raises ValueError if invalid.
    """

    if not isinstance(obj, dict):
        raise ValueError("Root must be a JSON object")

    if obj.get("type") != "questions":
        raise ValueError("Root must contain: type='questions'")

    if "title" not in obj or not isinstance(obj["title"], str):
        raise ValueError("Missing or invalid 'title'")

    if "questions" not in obj or not isinstance(obj["questions"], list):
        raise ValueError("Missing or invalid 'questions' list")

    if not obj["questions"]:
        raise ValueError("Questions list is empty")

    for i, q in enumerate(obj["questions"], start=1):
        if not isinstance(q, dict):
            raise ValueError(f"Question {i} is not an object")

        if q.get("qtype") != "mcq":
            raise ValueError(f"Question {i}: qtype must be 'mcq'")

        if not isinstance(q.get("prompt"), str):
            raise ValueError(f"Question {i}: missing or invalid prompt")

        choices = q.get("choices")
        if not isinstance(choices, list) or len(choices) != 4:
            raise ValueError(f"Question {i}: must have exactly 4 choices")

        if not all(isinstance(c, str) for c in choices):
            raise ValueError(f"Question {i}: all choices must be strings")

        ci = q.get("correct_index")
        if not isinstance(ci, int) or not (0 <= ci <= 3):
            raise ValueError(f"Question {i}: correct_index must be 0..3")

def template_expects_json(tpl_system_prompt: str) -> bool:
    if not tpl_system_prompt:
        return False
    s = tpl_system_prompt.lower()
    return "json" in s and "only" in s

def template_ui_new():
    rows = list_templates()
    if not rows:
        st.info("No templates found in ChatTemplates.")
        return

    title_to_id = {r[1]: r[0] for r in rows}
    titles = list(title_to_id.keys())

    chosen_title = st.selectbox("Template", titles, index=0)
    tpl_id = title_to_id[chosen_title]
    tpl = load_template(tpl_id)
    if not tpl:
        st.warning("Template could not be loaded.")
        return

    st.caption(f"Model: `{tpl['model']}`  •  Updated: {tpl['updated_at']}")

    # Namespace widget keys per-template so states don't collide across templates
    key_prefix = f"tpl_{tpl_id}_"

    def clear_fields_for_template():
        for f in tpl["fields"]:
            k = f"{key_prefix}{f['name']}"
            ftype = f.get("type", "text")

            if ftype == "text" or ftype == "textarea":
                ss[k] = ""
            elif ftype == "number":
                ss[k] = 0
            elif ftype == "checkbox":
                ss[k] = False
            elif ftype == "select":
                # Set to first option if available, else empty
                options = f.get("options", [])
                ss[k] = options[0] if options else ""
            else:
                ss[k] = ""

    # ----------------------------
    # Render fields
    # ----------------------------
    values = {}
    for f in tpl["fields"]:
        values[f["name"]] = render_field(f, key_prefix=key_prefix)

    # ----------------------------
    # Build previews
    # ----------------------------
    preview_plain = fill_prompt(tpl["user_prompt"], values)
    preview_html = fill_prompt_with_highlight(tpl["user_prompt"], values, tpl["fields"])

    # ----------------------------
    # Preview UI
    # ----------------------------
    st.markdown("#### Preview")
    st.markdown(preview_html, unsafe_allow_html=True)

    # Missing fields logic
    miss = missing_vars(tpl["user_prompt"], values)
    send_disabled = len(miss) > 0

    # ----------------------------
    # Action buttons
    # ----------------------------
    c1, c2 = st.columns([1, 1])

    with c1:
        if st.button("🚀 Send to Chat", disabled=send_disabled):
            ss.pending_prompt = preview_plain

            # template overrides for the API call
            ss.template_system = tpl["system_prompt"]
            ss.template_params = tpl["params"]
            ss.template_model = tpl["model"]

            st.rerun()  # ✅ This one is NOT in a callback, so it's fine.

    with c2:
        st.button("♻️ Clear Fields", on_click=clear_fields_for_template)

@st.dialog("Save chat")
def save_chat():

    file_name = st.text_input("Save as:")

    if st.button("Submit"):
        if not file_name:
            st.warning("Please enter a file name.")
            return

        # Ensure .pdf extension
        if not file_name.lower().endswith(".pdf"):
            file_name += ".pdf"

        save_file = PDF_DIR / file_name

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.add_font("DejaVu", "", str(FONT_PATH), uni=True)
        pdf.set_font("DejaVu", "", 16)

        pdf.cell(200, 10, file_name, ln=True, align="C")

        # Add chat messages
        pdf.set_font("DejaVu", "", 12)

        for message in ss.messages:
            role = "You" if message["role"] == "user" else "ChatGPT"
            pdf.multi_cell(0, 10, f"{role}: {message['content']}")
            pdf.ln()

        pdf.output(str(save_file))

        st.toast("PDF saved", icon="📄")
        st.rerun()

@st.dialog("📤 Publish to Student App")
def publish_dialog():

    if "last_response" not in ss or not ss.last_response:
        st.warning("No generated content to publish yet.")
        if st.button("Close"):
            st.rerun()
        return

    st.markdown("### Publish the latest AI response")

    title = st.text_input("Title")
    subject = st.text_input("Subject", "Chemistry")
    grade = st.text_input("Grade", "9")
    content_type = st.selectbox("Type", ["questions", "notes", "practice", "explanation"])

    st.divider()

    with st.expander("🔍 Preview content"):
        if isinstance(ss.last_response, (dict, list)):
            st.json(ss.last_response)
        else:
            st.markdown(translate_latex(ss.last_response))

    c1, c2 = st.columns(2)

    with c1:
        if st.button("📤 Publish"):
            if not title.strip():
                st.error("Title is required.")
            else:
                publish_item(
                    title=title,
                    subject=subject,
                    grade=grade,
                    content_type=content_type,
                    content=ss.last_response
                )
                st.success("Published to Student App! 🎉")
                st.rerun()

    with c2:
        if st.button("Cancel"):
            st.rerun()

# Rendering prompts page

st.markdown("""
<style>
main {
padding-top: 0rem !important;
margin-top: 0rem !important;
}
/* Reduce top padding of main app container */
div.block-container {
    padding-top: 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.tpl-missing {
    background-color: #fff3b0;
    color: #7a5c00;
    padding: 2px 6px;
    border-radius: 6px;
    font-weight: 600;
}
.tpl-filled {
    color: #0b5ed7;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# OpenAI API key stored in Streamlit secrets
openAI_api_key = st.secrets["OPENAI_API_KEY"]
client = Client(api_key=openAI_api_key)

for k, v in DEFAULT_LLM_CONFIG.items():
    if k not in ss:
        ss[k] = v

if "llm_preset" not in ss:
    ss.llm_preset = "Precise (Math / Science)"

# Initialize chat history
if "messages" not in ss:
    ss.messages = []
    ss.template = None
    ss.template_list = []
    ss.prompt = None
    ss.show_template = False
    ss.user_template = None
    ss.index = None

if "pending_prompt" not in ss:
    ss.pending_prompt = None

if "template_system" not in ss:
    ss.template_system = None
if "template_params" not in ss:
    ss.template_params = None
if "template_model" not in ss:
    ss.template_model = None
if "last_response" not in ss:
    ss.last_response = None

with st.sidebar:

    st.subheader("🤖 Chat Engine")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾", help="Save Chat"):
            save_chat()  # 👈 THIS OPENS THE DIALOG

    with col2:
        if st.button("📤", help="Publish to Student App"):
            publish_dialog()

    with col3:
        if st.button("📖", help="New Chat"):
            ss.messages = []
            st.rerun()

    st.divider()

    # ---- Preset selector ----
    preset = st.selectbox(
        "Preset",
        list(MODEL_PRESETS.keys()),
        key="llm_preset"
    )

    if st.button("Apply Preset"):
        for k, v in MODEL_PRESETS[preset].items():
            ss[k] = v
        st.rerun()

    st.divider()

    # ---- Model selector ----
    ss.openai_model = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o"],   # extend later
        index=["gpt-4o-mini", "gpt-4o"].index(ss.openai_model)
        if ss.openai_model in ["gpt-4o-mini", "gpt-4o"] else 0
    )

    st.divider()

    # ---- Sampling controls ----
    ss.temperature = st.slider(
        "Temperature",
        0.0, 2.0, float(ss.temperature), 0.05,
        help="Creativity / randomness"
    )

    ss.top_p = st.slider(
        "Top-P",
        0.0, 1.0, float(ss.top_p), 0.05,
        help="Nucleus sampling cutoff"
    )

    ss.presence = st.slider(
        "Presence penalty",
        -2.0, 2.0, float(ss.presence), 0.1,
        help="Encourage new topics"
    )

    ss.frequency = st.slider(
        "Frequency penalty",
        -2.0, 2.0, float(ss.frequency), 0.1,
        help="Discourage repetition"
    )

    st.divider()

    # ---- Debug / inspection (optional but very useful) ----
    with st.expander("🔍 Current LLM Settings"):
        st.json({
            "model": ss.openai_model,
            "temperature": ss.temperature,
            "top_p": ss.top_p,
            "presence": ss.presence,
            "frequency": ss.frequency,
        })

st.header("Prompts")

with st.expander("📄 Templates — build a prompt", expanded=False):
    template_ui_new()

# Render messages
for message in ss.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("What do you want to know?")
if user_prompt:
    ss.pending_prompt = user_prompt

if ss.pending_prompt:
    prompt_to_send = ss.pending_prompt
    ss.pending_prompt = None

    # Add user message to chat history
    ss.messages.append({"role": "user", "content": prompt_to_send})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt_to_send)

    with st.chat_message("assistant"):
        response_container = st.empty()

        tpl_params = ss.template_params or {}

        model = ss.template_model or ss.openai_model
        temperature = tpl_params.get("temperature", ss.temperature)
        top_p = tpl_params.get("top_p", ss.top_p)
        presence_penalty = tpl_params.get("presence", ss.presence)
        frequency_penalty = tpl_params.get("frequency", ss.frequency)

        system_prompt = ss.template_system or (
            "When you include math or chemistry formulas, ALWAYS wrap inline math in $...$ "
            "and display equations in $$...$$. Never output bare LaTeX commands without delimiters."
        )

        stream = client.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            messages=[
                         {"role": "system", "content": system_prompt}
                     ] + [{"role": m["role"], "content": m["content"]} for m in ss.messages],
            stream=True,
            stream_options={"include_usage": True},
        )

        response = ""

        for chunk in stream:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    response += delta.content
                    response_container.markdown(translate_latex(response))

        # Final render safety
        response_container.markdown(translate_latex(response))

    ss.messages.append({"role": "assistant", "content": response})

    expects_json = template_expects_json(ss.template_system)

    # Default: treat as normal text
    ss.last_response = response

    if expects_json:
        json_ok = True

        # ---------- TRY PARSE ----------
        try:
            parsed = json.loads(response)
        except Exception as e:
            st.error("❌ Model did not return valid JSON.")
            st.code(response)
            ss.last_response = None
            json_ok = False

        # ---------- TRY SCHEMA ----------
        if json_ok:
            try:
                if isinstance(parsed, dict) and parsed.get("type") == "questions":
                    validate_questions_schema(parsed)
            except Exception as e:
                st.error(f"❌ JSON schema invalid: {e}")
                st.json(parsed)
                ss.last_response = None
                json_ok = False

        # ---------- SUCCESS ----------
        if json_ok:
            ss.last_response = parsed
            st.success("✅ Valid worksheet JSON generated.")

    # Clear template overrides
    ss.template_system = None
    ss.template_params = None
    ss.template_model = None

