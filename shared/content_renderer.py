import streamlit as st
from streamlit import session_state as ss
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
                key=f"q_{qid}",
                index=None
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
                if user_ans is None:
                    pass  # unanswered
                else:
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

def render_published_content(content: str):
    content_str = content.strip()

    if content_str.startswith("{"):
        try:
            data = json.loads(content_str)

            if isinstance(data, dict) and data.get("type") == "questions":
                render_interactive_questions(data)
            else:
                st.json(data)

        except Exception:
            st.error("This content looks like JSON but could not be parsed.")
            st.markdown(translate_latex(content), unsafe_allow_html=True)
    else:
        st.markdown(translate_latex(content), unsafe_allow_html=True)
