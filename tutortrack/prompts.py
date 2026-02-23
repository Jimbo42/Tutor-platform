import io
import re
import sqlite3
from openai import Client
import streamlit as st
from streamlit import session_state as ss
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import json
from datetime import datetime
from pathlib import Path

from tutortrack.lessons import get_conn
from shared.google_db import publish_item
from shared.content_renderer import render_interactive_questions, translate_latex
from shared.google_db import (
    upload_interactive_json,
    upload_pdf_bytes,
)

# Tutor root
ROOT = Path(__file__).resolve().parents[1]

# Shared PDF output folder
PDF_DIR = ROOT / "shared" / "pdf_files"
PDF_DIR.mkdir(parents=True, exist_ok=True)

# Font path
FONT_PATH = ROOT / "tutortrack" / "resources" / "fonts" / "DejaVuSans.ttf"
FONT_BOLD_PATH = ROOT / "tutortrack" / "resources" / "fonts" / "DejaVuSans-Bold.ttf"

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

def is_interactive_response(val) -> bool:
    return isinstance(val, (dict, list))

def is_notes_response(val) -> bool:
    return isinstance(val, str) and val.strip() != ""

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

def _chat_to_editable_text(keep_prompts: bool = False) -> str:
    """
    Build an editable plain-text version of the chat.
    If keep_prompts=False, only include assistant messages.
    """
    lines = []
    for m in ss.get("messages", []):
        if keep_prompts:
            role = "You" if m["role"] == "user" else "ChatGPT"
            lines.append(f"{role}: {m['content']}".strip())
        else:
            if m["role"] == "assistant":
                lines.append(m["content"].strip())
    return "\n\n".join([x for x in lines if x])

def make_sharp_pdf_bytes(
    *,
    title: str,
    subject: str,
    grade: str,
    body: str,
    font_size: int = 12,
    line_spacing: str = "normal",  # "tight" | "normal"
    page_numbers: bool = True,
) -> bytes:
    import re
    from datetime import datetime
    from fpdf import FPDF

    # ----------------------------
    # Patterns
    # ----------------------------
    BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
    MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
    URL_RE = re.compile(r"(https?://\S+)")
    BLOCK_MATH_RE = re.compile(r"^\s*\$\$(.*?)\$\$\s*$")

    # Subscript map for digits
    SUB_MAP = str.maketrans({
        "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
        "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉"
    })

    # ----------------------------
    # PDF class (footer)
    # ----------------------------
    class PDF(FPDF):
        def footer(self):
            if not page_numbers:
                return
            self.set_y(-12)
            self.set_font("DejaVu", "", 9)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    # ----------------------------
    # Text helpers
    # ----------------------------
    def _normalize(s: str) -> str:
        if s is None:
            return ""
        s = str(s).replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"[\x00-\x08\x0b-\x1f]", "", s)  # strip control chars
        s = s.replace("\u00A0", " ")               # NBSP -> space
        return s

    def _break_long_tokens(line: str, chunk: int = 90) -> str:
        parts = []
        for tok in re.split(r"(\s+)", line):
            if tok.strip() and len(tok) > chunk and not tok.isspace():
                parts.append(" ".join(tok[i:i + chunk] for i in range(0, len(tok), chunk)))
            else:
                parts.append(tok)
        return "".join(parts)

    def latex_to_pretty_text(s: str) -> str:
        """Small LaTeX->text pass for common biology/chemistry patterns."""
        s = (s or "").strip()

        # strip $$...$$ if present
        s = s.replace("$$", "").strip()

        # \text{...} -> ...
        s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)

        # arrows
        s = s.replace("\\rightarrow", "→").replace("\\to", "→")

        # subscripts _{digits} and _digits
        s = re.sub(r"_\{(\d+)\}", lambda m: m.group(1).translate(SUB_MAP), s)
        s = re.sub(r"_(\d+)", lambda m: m.group(1).translate(SUB_MAP), s)

        # remove braces
        s = s.replace("{", "").replace("}", "")

        # normalize spacing
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # ----------------------------
    # Bold rendering helpers
    # ----------------------------
    def _parse_bold_runs(text: str):
        """Return list of (style, chunk) where style is 'normal' or 'bold'."""
        runs = []
        i = 0
        for m in BOLD_RE.finditer(text):
            if m.start() > i:
                runs.append(("normal", text[i:m.start()]))
            runs.append(("bold", m.group(1)))
            i = m.end()
        if i < len(text):
            runs.append(("normal", text[i:]))
        return runs or [("normal", text)]

    def _tokenize_runs(runs):
        """Yield (style, token) preserving whitespace tokens."""
        for style, chunk in runs:
            for tok in re.split(r"(\s+)", chunk):
                if tok == "":
                    continue
                yield style, tok

    def _set_style(style: str):
        pdf.set_font("DejaVu", "B" if style == "bold" else "", font_size)

    def _tok_width(style: str, tok: str) -> float:
        _set_style(style)
        return pdf.get_string_width(tok)

    def _write_bold_wrapped_line(line_text: str):
        """
        Write one logical line with **bold** support, wrapped.
        Uses pdf.write so we can mix fonts on a single line.
        """
        runs = _parse_bold_runs(line_text)
        tokens = list(_tokenize_runs(runs))
        if not tokens:
            pdf.ln(lh)
            return

        current = []
        cur_w = 0.0

        def flush():
            nonlocal current, cur_w
            pdf.set_x(pdf.l_margin)
            for style, tok in current:
                _set_style(style)
                pdf.write(lh, tok)
            pdf.ln(lh)
            current = []
            cur_w = 0.0

        for style, tok in tokens:
            w = _tok_width(style, tok)

            if cur_w + w > max_w and cur_w > 0:
                flush()

            # hard split over-wide tokens
            if w > max_w:
                step = max(1, int(len(tok) * (max_w / w)))
                for j in range(0, len(tok), step):
                    piece = tok[j:j + step]
                    pw = _tok_width(style, piece)
                    if cur_w + pw > max_w and cur_w > 0:
                        flush()
                    current.append((style, piece))
                    cur_w += pw
                continue

            current.append((style, tok))
            cur_w += w

        if current:
            flush()

    # ----------------------------
    # Clickable link writer
    # ----------------------------
    def write_markdown_links_line(text: str):
        """
        Writes one logical line.
        Supports:
          - Markdown links: [label](url) -> clickable label
          - bare URLs -> clickable URL
        """
        pdf.set_x(pdf.l_margin)

        # Split into text/link segments first
        segments = []
        i = 0
        for m in MD_LINK_RE.finditer(text):
            if m.start() > i:
                segments.append(("text", text[i:m.start()], None))
            segments.append(("link", m.group(1), m.group(2)))
            i = m.end()
        if i < len(text):
            segments.append(("text", text[i:], None))

        # Expand text segments into text + bare URL segments
        expanded = []
        for kind, chunk, url in segments:
            if kind == "link":
                expanded.append((kind, chunk, url))
            else:
                last = 0
                for u in URL_RE.finditer(chunk):
                    if u.start() > last:
                        expanded.append(("text", chunk[last:u.start()], None))
                    expanded.append(("url", u.group(1), u.group(1)))
                    last = u.end()
                if last < len(chunk):
                    expanded.append(("text", chunk[last:], None))

        cur_w = 0.0

        def flush_line():
            nonlocal cur_w
            pdf.ln(lh)
            pdf.set_x(pdf.l_margin)
            cur_w = 0.0

        for kind, chunk, url in expanded:
            if not chunk:
                continue

            pdf.set_font("DejaVu", "", font_size)
            w = pdf.get_string_width(chunk)

            if cur_w + w > max_w and cur_w > 0:
                flush_line()

            if w > max_w:
                # hard split
                step = max(1, int(len(chunk) * (max_w / w)))
                for j in range(0, len(chunk), step):
                    piece = chunk[j:j + step]
                    pdf.set_font("DejaVu", "", font_size)
                    pw = pdf.get_string_width(piece)
                    if cur_w + pw > max_w and cur_w > 0:
                        flush_line()
                    if kind in ("link", "url"):
                        pdf.set_text_color(0, 0, 255)
                        pdf.write(lh, piece, link=url)
                        pdf.set_text_color(0, 0, 0)
                    else:
                        pdf.write(lh, piece)
                    cur_w += pw
                continue

            if kind in ("link", "url"):
                pdf.set_text_color(0, 0, 255)
                pdf.write(lh, chunk, link=url)
                pdf.set_text_color(0, 0, 0)
            else:
                pdf.write(lh, chunk)

            cur_w += w

        pdf.ln(lh)

    # ----------------------------
    # Normalize inputs
    # ----------------------------
    title = _normalize(title).strip()
    subject = _normalize(subject).strip()
    grade = _normalize(grade).strip()
    body = _normalize(body)

    # ----------------------------
    # Build PDF
    # ----------------------------
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(18, 16, 18)
    pdf.add_page()

    # Fonts (Unicode-safe)
    pdf.add_font("DejaVu", "", str(FONT_PATH))
    pdf.add_font("DejaVu", "B", str(FONT_BOLD_PATH))

    # Title
    pdf.set_font("DejaVu", "", 18)
    pdf.cell(0, 10, title or "Untitled", new_x="LMARGIN", new_y="NEXT", align="C")

    # Meta line
    pdf.set_font("DejaVu", "", 11)
    meta = "   •   ".join(
        [x for x in [
            f"Subject: {subject}" if subject else "",
            f"Grade: {grade}" if grade else "",
            datetime.now().strftime("%Y-%m-%d"),
        ] if x]
    )
    pdf.cell(0, 7, meta, new_x="LMARGIN", new_y="NEXT", align="C")

    # Divider
    pdf.ln(2)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(6)

    # Body settings
    max_w = pdf.w - pdf.l_margin - pdf.r_margin
    lh = 6 if line_spacing == "tight" else 7

    # Render body line-by-line
    for raw_line in body.split("\n"):
        safe_line = _break_long_tokens(raw_line, chunk=90)

        # (1) Convert $$...$$ single-line math blocks to readable text
        m = BLOCK_MATH_RE.match(safe_line.strip())
        if m:
            pretty = latex_to_pretty_text(m.group(1))
            pdf.set_font("DejaVu", "", font_size)
            pdf.set_x(pdf.l_margin)
            try:
                pdf.multi_cell(max_w, lh, pretty, wrapmode="CHAR")
            except TypeError:
                pdf.multi_cell(max_w, lh, pretty)
            continue

        # (2) Links: markdown links + bare URLs (clickable)
        if ("http://" in safe_line) or ("https://" in safe_line) or ("[" in safe_line and "](" in safe_line):
            # If line ALSO contains **bold**, strip markers for now
            # (we can combine bold+links next if you want)
            safe_links = safe_line.replace("**", "")
            write_markdown_links_line(safe_links)
            continue

        # (3) Bold: **bold** support
        if "**" in safe_line:
            try:
                _write_bold_wrapped_line(safe_line)
            except Exception:
                # fallback plain, strip markers
                plain = safe_line.replace("**", "")
                pdf.set_font("DejaVu", "", font_size)
                pdf.set_x(pdf.l_margin)
                try:
                    pdf.multi_cell(max_w, lh, plain, wrapmode="CHAR")
                except TypeError:
                    pdf.multi_cell(max_w, lh, plain)
            continue

        # (4) Plain text
        pdf.set_font("DejaVu", "", font_size)
        pdf.set_x(pdf.l_margin)
        try:
            pdf.multi_cell(max_w, lh, safe_line, wrapmode="CHAR")
        except TypeError:
            pdf.multi_cell(max_w, lh, safe_line)

    out = pdf.output()  # bytes / bytearray in fpdf2
    return bytes(out) if not isinstance(out, bytes) else out

@st.dialog("Edit / Save", width="large")
def edit_save_dialog():
    st.markdown("### Save or publish")

    # Detect the latest output type
    last = ss.get("last_response", None)
    is_interactive = ss.get("last_output_kind") == "interactive" and is_interactive_response(last)
    is_notes = is_notes_response(last) or isinstance(last, str)

    # -------- Metadata (shared)
    c1, c2, c3 = st.columns([2, 2, 1.5])
    with c1:
        # If interactive JSON includes its own title, prefer it as default
        default_title = "Study Notes"
        if is_interactive and isinstance(last, dict) and isinstance(last.get("title"), str) and last["title"].strip():
            default_title = last["title"].strip()
        title = st.text_input("Title", value=default_title)

    with c2:
        subject = st.text_input("Subject", value="Math")
    with c3:
        grade = st.text_input("Grade", value="All")

    make_public_flag = st.checkbox("Make file accessible to anyone with the link", value=True)

    st.divider()

    # ============================
    # ✅ Interactive publish mode
    # ============================
    if is_interactive:
        st.caption("Detected: **Interactive (JSON)** — this will publish to the Interactives folder.")

        # Optional: show a small summary (no markdown preview)
        if isinstance(last, dict):
            qn = len(last.get("questions", [])) if isinstance(last.get("questions"), list) else None
            if qn is not None:
                st.info(f"Questions detected: {qn}")

        b1, b2 = st.columns([1, 1])

        with b1:
            # Download JSON locally (handy for debugging)
            if st.button("💾 Save Local (Download JSON)", use_container_width=True):
                if not title.strip():
                    st.error("Title is required.")
                    return
                st.download_button(
                    "⬇️ Download now",
                    data=json.dumps(last, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name=f"{title.strip()}.json",
                    mime="application/json",
                    use_container_width=True,
                )

        with b2:
            if st.button("📤 Publish Interactive to Student", use_container_width=True):
                if not title.strip():
                    st.error("Title is required.")
                    return

                # Upload JSON to Drive (Interactives folder)
                payload = upload_interactive_json(
                    obj=last,
                    title=title.strip(),
                    make_public=make_public_flag,
                )

                # Add row to PublishedItems DB
                publish_item(
                    title=title.strip(),
                    subject=subject.strip(),
                    grade=grade.strip(),
                    content_type="interactive",
                    content=payload,
                )

                st.success("Published Interactive to Student Library ✅")
                st.rerun()

        return  # ✅ stop here for interactive mode

    # ============================
    # 📝 Notes/PDF mode (existing)
    # ============================
    st.caption("Detected: **Notes (text)** — this will save/publish a PDF.")

    # -------- Formatting controls
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1.4])
    with f1:
        keep_prompts = st.checkbox("Include prompts", value=False)
    with f2:
        font_size = st.selectbox("Font size", [11, 12, 13], index=1)
    with f3:
        line_spacing = st.selectbox("Line spacing", ["Tight", "Normal"], index=1)
    with f4:
        page_numbers = st.checkbox("Page numbers", value=True)

    # Build default text to edit
    default_text = _chat_to_editable_text(keep_prompts=keep_prompts)

    # -------- Split view: editor + preview
    left, right = st.columns([1.15, 1], vertical_alignment="top")

    with left:
        edited = st.text_area("Content", value=default_text, height=420)

    with right:
        st.markdown("**Preview (Markdown-style)**")
        st.markdown(
            f"#### {title.strip() or 'Untitled'}\n"
            f"*Subject:* {subject.strip() or '-'}  \n"
            f"*Grade:* {grade.strip() or '-'}\n\n"
            f"---\n\n"
            f"{edited[:4000] if edited else ''}"
        )
        if edited and len(edited) > 4000:
            st.caption("Preview truncated… PDF will include full text.")

    st.divider()

    # ---- Save / Publish PDF
    b1, b2 = st.columns([1, 1])

    with b1:
        if st.button("💾 Save Local (Download PDF)", use_container_width=True):
            if not title.strip():
                st.error("Title is required.")
                return
            pdf_bytes = make_sharp_pdf_bytes(
                title=title.strip(),
                subject=subject.strip(),
                grade=grade.strip(),
                body=edited,
                font_size=int(font_size),
                line_spacing=("tight" if line_spacing == "Tight" else "normal"),
                page_numbers=bool(page_numbers),
            )
            st.download_button(
                "⬇️ Download now",
                data=pdf_bytes,
                file_name=f"{title.strip()}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    with b2:
        if st.button("📤 Publish PDF to Student", use_container_width=True):
            if not title.strip():
                st.error("Title is required.")
                return

            pdf_bytes = make_sharp_pdf_bytes(
                title=title.strip(),
                subject=subject.strip(),
                grade=grade.strip(),
                body=edited,
                font_size=int(font_size),
                line_spacing=("tight" if line_spacing == "Tight" else "normal"),
                page_numbers=bool(page_numbers),
            )

            payload = upload_pdf_bytes(
                pdf_bytes=pdf_bytes,
                title=title.strip(),
                make_public=make_public_flag,
            )

            publish_item(
                title=title.strip(),
                subject=subject.strip(),
                grade=grade.strip(),
                content_type="pdf",
                content=payload,
            )

            st.success("Published PDF to Student Library ✅")
            st.rerun()

def render_latest_preview():
    if not ss.get("last_response"):
        return

    st.markdown("## ✅ Latest Output Preview")

    with st.container(border=True):

        if is_interactive_response(ss.last_response):
            st.caption("Detected: **Interactive (JSON)**")
            render_interactive_questions(ss.last_response)
            # st.json(ss.last_response)
        else:
            st.caption("Detected: **Notes (text)**")
            st.markdown(translate_latex(str(ss.last_response)))

    st.divider()

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
if "last_json_parsed" not in ss:
    ss.last_json_parsed = None
if "last_output_kind" not in ss:
    ss.last_output_kind = "unknown"

with st.sidebar:

    st.subheader("🤖 Chat Engine")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✏️ Edit/Save", use_container_width=True):
            edit_save_dialog()

    with col2:
        if st.button("🆕 New Chat", use_container_width=True):
            ss.messages = []
            ss.last_response = None
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

    # ---- Decide whether we should attempt JSON parsing ----
    # Prefer template_system when present, but also allow "response looks like JSON" fallback
    expects_json = template_expects_json(ss.template_system)


    def looks_like_json(s: str) -> bool:
        s = (s or "").lstrip()
        return s.startswith("{") or s.startswith("[")


    should_try_json = expects_json or looks_like_json(response)

    # Default: store raw response as notes
    ss.last_response = response
    ss.last_output_kind = "notes"

    if should_try_json:
        parsed = None
        parsed_ok = False
        schema_ok = False

        # ---------- TRY PARSE ----------
        try:
            parsed = json.loads(response)
            parsed_ok = isinstance(parsed, (dict, list))
        except Exception:
            st.error("❌ Model did not return valid JSON.")
            st.code(response)

        # Always keep the parsed object if we got one (even if schema fails)
        if parsed_ok:
            ss.last_json_parsed = parsed

            # ---------- TRY SCHEMA (only if it is the worksheet type) ----------
            try:
                if isinstance(parsed, dict) and parsed.get("type") == "questions":
                    validate_questions_schema(parsed)
                    schema_ok = True
                else:
                    # Non-worksheet JSON is still "interactive-ish" for publishing
                    schema_ok = True
            except Exception as e:
                st.error(f"❌ JSON schema invalid: {e}")
                st.json(parsed)
                schema_ok = False

            # If parse succeeded, treat output as interactive for Edit/Save
            ss.last_output_kind = "interactive"
            ss.last_response = parsed  # so your existing preview/edit logic can use it

            if schema_ok:
                st.success("✅ Valid worksheet JSON generated.")
            else:
                st.warning("⚠️ JSON parsed, but schema failed. You can still publish or download and fix it.")

            render_latest_preview()
        else:
            # Parse failed; keep as notes (raw text)
            ss.last_output_kind = "notes"

    # Clear template overrides
    ss.template_system = None
    ss.template_params = None
    ss.template_model = None

