import streamlit as st
from streamlit import session_state as ss
import json
import re
import streamlit.components.v1 as components
import requests

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

def _render_gdrive_pdf(payload: dict):
    preview_url = payload.get("preview_url") or payload.get("view_url")
    if not preview_url:
        st.error("PDF payload missing preview_url/view_url")
        st.json(payload)
        return

    # Best embedding endpoint for Drive PDFs
    if "/preview" not in preview_url and "/file/d/" in preview_url:
        preview_url = preview_url.replace("/view", "/preview")

    components.iframe(preview_url, height=900, scrolling=True)

    view_url = payload.get("view_url")
    if view_url:
        st.link_button("Open in Google Drive", view_url)

import requests

def _render_gdrive_interactive(payload: dict, key_prefix: str = ""):
    download_url = payload.get("download_url")
    view_url = payload.get("view_url")

    st.info("Interactive content stored in Google Drive.")

    if payload.get("filename"):
        st.caption(f"File: **{payload['filename']}**")

    # Buttons
    c1, c2 = st.columns(2)
    with c1:
        if view_url:
            st.link_button("Open in Google Drive", view_url)
    with c2:
        if download_url:
            st.link_button("Download JSON", download_url)

    if not download_url:
        st.warning("No download URL found.")
        return

    # ---- AUTO FETCH + CACHE ----
    cache_key = f"{key_prefix}interactive_json"

    if cache_key not in ss:
        try:
            r = requests.get(download_url, timeout=15)
            r.raise_for_status()
            ss[cache_key] = r.json()
        except Exception as e:
            st.error(f"Could not load JSON from Google Drive: {e}")
            with st.expander("Payload", expanded=True):
                st.json(payload)
            return

    data = ss[cache_key]

    # ---- STUDENT EXPERIENCE RENDER ----
    if isinstance(data, dict) and data.get("type") == "questions":
        st.divider()
        st.subheader("Student Preview")
        render_interactive_questions(data)
    else:
        st.warning("Downloaded JSON is not a worksheet.")
        st.json(data)

    # Optional payload debug
    with st.expander("Payload", expanded=False):
        st.json(payload)

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

def render_published_content(content: str, content_type: str | None = None):
    """
    content: TEXT pulled from SQLite (either raw JSON worksheet, plain text, or a Drive payload JSON)
    content_type: optional, from PublishedItems.content_type
      - "pdf" -> expect Drive payload and embed PDF
      - "interactive" -> expect Drive payload and show links
      - otherwise: auto-detect questions JSON vs plain text
    """
    content_str = (content or "").strip()

    # ---- type-directed rendering first ----
    if content_type in ("pdf", "interactive"):
        if not content_str.startswith("{"):
            st.error(f"Expected JSON payload for content_type={content_type}, got plain text.")
            st.markdown(translate_latex(content_str), unsafe_allow_html=True)
            return

        try:
            payload = json.loads(content_str)
        except Exception:
            st.error("Could not parse JSON payload.")
            st.code(content_str)
            return

        provider = payload.get("provider")
        if provider == "gdrive":
            if content_type == "pdf":
                _render_gdrive_pdf(payload)
                return
            if content_type == "interactive":
#                _render_gdrive_interactive(payload)
                _render_gdrive_interactive(payload, key_prefix=f"pm_{payload.get('file_id', '')}_")

                return

        # fallback for unknown provider
        st.warning("Unknown provider payload; showing raw JSON.")
        st.json(payload)
        return

    # ---- legacy auto-detect mode ----
    if content_str.startswith("{"):
        try:
            data = json.loads(content_str)

            # If it's a worksheet JSON, render interactively
            if isinstance(data, dict) and data.get("type") == "questions":
                render_interactive_questions(data)
            else:
                st.json(data)

        except Exception:
            st.error("This content looks like JSON but could not be parsed.")
            st.markdown(translate_latex(content_str), unsafe_allow_html=True)
    else:
        st.markdown(translate_latex(content_str), unsafe_allow_html=True)
