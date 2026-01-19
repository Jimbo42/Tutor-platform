import streamlit as st
from streamlit import session_state as ss
from shared.published_db import get_published_items, get_published_item_by_id
import json
import re

_MATH_BLOCK_OR_INLINE = re.compile(r"(\$\$.*?\$\$|\$.*?\$)", re.DOTALL)

_TEX_RUN = re.compile(
    r"(?:\\[A-Za-z]+(?:\{[^}]*\})*[A-Za-z0-9{}\[\]_^%:+\-*/().=,]*)+"
)

def translate_latex(val: str) -> str:
    if "\\" not in val and "$" not in val:
        return val

    s = val

    # Normalize \( \) and \[ \] into $ and $$
    s = re.sub(r"\\\((.*?)\\\)", r"$\1$", s, flags=re.DOTALL)
    s = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", s, flags=re.DOTALL)

    parts = _MATH_BLOCK_OR_INLINE.split(s)

    def wrap_tex_runs(text: str) -> str:
        def repl(m: re.Match) -> str:
            expr = m.group(0)
            m2 = re.match(r"^(.*?)([.,;:!?)]*)$", expr)
            core, punct = m2.group(1), m2.group(2)
            return f"${core}$" + punct

        return _TEX_RUN.sub(repl, text)

    for i in range(len(parts)):
        if i % 2 == 0:
            parts[i] = wrap_tex_runs(parts[i])

    return "".join(parts)

def render_interactive_questions(data):
    st.subheader(data.get("title", "Practice Questions"))

    if "answers_checked" not in st.session_state:
        ss.answers_checked = False

    questions = data.get("questions", [])

    # ----------------------------
    # Render questions
    # ----------------------------
    for q in questions:
        qid = q["id"]
        qtype = q.get("qtype", "short")

        st.markdown(f"### {qid}. {q['prompt']}")

        # Student input
        if qtype == "short":
            user_ans = st.text_input(
                "Your answer:",
                key=f"q_{qid}"
            )

        elif qtype == "mcq":
            choices = q["choices"]
            user_ans = st.radio(
                "Choose one:",
                choices,
                key=f"q_{qid}"
            )

        # ----------------------------
        # After checking: show result
        # ----------------------------
        if ss.answers_checked:
            if qtype == "short":
                correct = q.get("answer", "")

                if short_answer_is_correct(user_ans, q):
                    st.success("✅ Correct")
                else:
                    st.error("❌ Incorrect")
                    st.markdown(f"**Correct answer:** {correct}")

                    # Optional: show guidance if keywords exist
                    if "keywords" in q:
                        st.info("Key ideas: " + ", ".join(q["keywords"]))

            elif qtype == "mcq":
                correct_choice = q["choices"][q["correct_index"]]

                if user_ans == correct_choice:
                    st.success("✅ Correct")
                else:
                    st.error("❌ Incorrect")
                    st.markdown(f"**Correct answer:** {correct_choice}")

        st.divider()

    # ----------------------------
    # Check answers button
    # ----------------------------
    if st.button("✅ Check Answers"):
        ss.answers_checked = True
        st.rerun()

    # ----------------------------
    # Score summary
    # ----------------------------
    if ss.answers_checked:
        score = 0
        total = len(questions)

        for q in questions:
            qid = q["id"]
            qtype = q.get("qtype", "short")
            user_ans = ss.get(f"q_{qid}")

            if qtype == "short":
                if short_answer_is_correct(user_ans, q):
                    score += 1

            elif qtype == "mcq":
                correct_choice = q["choices"][q["correct_index"]]
                if user_ans == correct_choice:
                    score += 1

        st.success(f"🎯 **Score: {score} / {total}**")

    # ----------------------------
    # Try again
    # ----------------------------
    if ss.answers_checked:
        if st.button("🔄 Try Again"):
            ss.answers_checked = False
            for q in questions:
                k = f"q_{q['id']}"
                if k in ss:
                    del ss[k]
            st.rerun()

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def short_answer_is_correct(user_ans: str, q: dict) -> bool:
    if not user_ans:
        return False

    user = normalize_text(user_ans)

    # 1) Exact / near-exact match against accept list
    accept_list = q.get("accept")
    if accept_list:
        for a in accept_list:
            if normalize_text(a) in user or user in normalize_text(a):
                return True

    # 2) Keyword-based scoring
    keywords = q.get("keywords", [])
    if keywords:
        hits = 0
        for kw in keywords:
            if normalize_text(kw) in user:
                hits += 1

        # Require at least 50% of keywords
        if hits >= max(1, len(keywords) // 2):
            return True

    # 3) Fallback: exact match with "answer"
    answer = q.get("answer", "")
    if normalize_text(answer) == user:
        return True

    return False

st.title("📚 Practice Library")

# ----------------------------
# Load items
# ----------------------------
items = get_published_items()

if not items:
    st.info("No published items yet.")
    st.stop()

# items rows:
# (id, title, subject, grade, content_type, created_at)

# Convert to list of dicts for easier handling
records = []
for r in items:
    records.append({
        "id": r[0],
        "title": r[1],
        "subject": r[2] or "Other",
        "grade": r[3] or "Other",
        "type": r[4] or "Other",
        "created": r[5],
    })

# ----------------------------
# Filters
# ----------------------------
st.subheader("🔎 Filter")

subjects = sorted(set(r["subject"] for r in records))
grades   = sorted(set(r["grade"] for r in records))
types    = sorted(set(r["type"] for r in records))

c1, c2, c3 = st.columns(3)

with c1:
    f_subject = st.selectbox("Subject", ["All"] + subjects)
with c2:
    f_grade = st.selectbox("Grade", ["All"] + grades)
with c3:
    f_type = st.selectbox("Type", ["All"] + types)

def passes_filters(r):
    if f_subject != "All" and r["subject"] != f_subject:
        return False
    if f_grade != "All" and r["grade"] != f_grade:
        return False
    if f_type != "All" and r["type"] != f_type:
        return False
    return True

filtered = [r for r in records if passes_filters(r)]

if not filtered:
    st.warning("No items match these filters.")
    st.stop()

st.divider()

# ----------------------------
# Direct link support
# ----------------------------
qp = st.query_params
forced_id = qp.get("item")

# ----------------------------
# Build selector
# ----------------------------
label_map = {}
labels = []

for r in filtered:
    label = f"{r['title']}  —  {r['subject']} • Grade {r['grade']} • {r['type']}"
    labels.append(label)
    label_map[label] = r["id"]

# Pick default
default_index = 0
if forced_id:
    try:
        forced_id = int(forced_id)
        for i, r in enumerate(filtered):
            if r["id"] == forced_id:
                default_index = i
                break
    except:
        pass

# Add placeholder option
labels_with_placeholder = ["— Select an item —"] + labels

# If URL forced_id exists, try to preselect it
default_index = 0

if forced_id:
    try:
        forced_id = int(forced_id)
        for i, r in enumerate(filtered):
            if r["id"] == forced_id:
                default_index = labels.index(
                    f"{r['title']}  —  {r['subject']} • Grade {r['grade']} • {r['type']}"
                ) + 1  # +1 because of placeholder
                break
    except:
        pass

choice = st.selectbox(
    "Choose an item:",
    labels_with_placeholder,
    index=default_index
)

# If still on placeholder → stop
if choice == "— Select an item —":
    st.info("👆 Select a worksheet to begin.")
    st.stop()

item_id = label_map[choice]

# Update URL (so it can be shared)
st.query_params["item"] = str(item_id)

# ----------------------------
# Load item
# ----------------------------
item = get_published_item_by_id(item_id)

st.divider()

_, title, subject, grade, ctype, content, created = item

st.subheader(title)
st.caption(f"{subject} • Grade {grade} • {ctype}")

# ----------------------------
# Render content
# ----------------------------
content_str = content.strip()

if content_str.startswith("{"):
    try:
        data = json.loads(content_str)

        if isinstance(data, dict) and data.get("type") == "questions":
            render_interactive_questions(data)
        else:
            st.json(data)

    except Exception as e:
        st.error("This content looks like JSON but could not be parsed.")
        st.markdown(translate_latex(content), unsafe_allow_html=True)

else:
    st.markdown(translate_latex(content), unsafe_allow_html=True)
