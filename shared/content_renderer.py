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
#        render_interactive_questions_paged(data, key_prefix=f"{key_prefix}q_")
        render_interactive_questions(data)
    else:
        st.warning("Downloaded JSON is not a worksheet.")
        st.json(data)

    # Optional payload debug
    with st.expander("Payload", expanded=False):
        st.json(payload)

def render_interactive_questions(data, *, ws_key: str = ""):
    """
    Robust one-question-at-a-time renderer.

    Key idea:
    - NEVER rely on widget session_state as the authoritative source of answers.
    - Store all submitted answers in ONE non-widget dict:
        ss[f"ws_{scope}_state"]["answers"][i] = answer
        ss[f"ws_{scope}_state"]["answered"][i] = True
    - Widgets are just input controls; navigation and scoring read from state["answers"].
    """
    st.subheader(data.get("title", "Practice Questions"))

    questions = data.get("questions", [])
    if not isinstance(questions, list) or not questions:
        st.warning("No questions found.")
        return

    # Worksheet scope
    title_key = normalize_text(data.get("title", "worksheet"))[:30] or "worksheet"
    scope = (normalize_text(ws_key)[:40] if ws_key else title_key) or title_key

    state_key = f"ws_{scope}_state"

    # state = { "idx": int, "answers": {i: val}, "answered": {i: True}, "review": bool }
    if state_key not in ss or not isinstance(ss.get(state_key), dict):
        ss[state_key] = {"idx": 0, "answers": {}, "answered": {}, "review": False}

    state = ss[state_key]

    def _q_meta(q: dict, i: int):
        q = q if isinstance(q, dict) else {}
        qid = q.get("id", i + 1)
        qtype = (q.get("qtype") or ("mcq" if "choices" in q else "short")).strip().lower()
        prompt = q.get("prompt") or q.get("question") or q.get("text") or f"Question {i+1}"
        return q, qid, qtype, prompt

    def _clamp_idx():
        i0 = int(state.get("idx", 0) or 0)
        i0 = max(0, min(i0, len(questions) - 1))
        state["idx"] = i0
        ss[state_key] = state
        return i0

    def _reset_all():
        ss[state_key] = {"idx": 0, "answers": {}, "answered": {}, "review": False}
        # Also clear any old widget keys for this scope (optional but helps cleanliness)
        for k in list(ss.keys()):
            if k.startswith(f"ws_{scope}_w_"):
                del ss[k]

    def _is_answered(i: int) -> bool:
        return bool(state.get("answered", {}).get(i, False))

    def _get_answer(i: int):
        return state.get("answers", {}).get(i, None)

    def _has_value(v):
        if v is None:
            return False
        if isinstance(v, str):
            return v.strip() != ""
        return True

    def _compute_correct(i: int) -> bool:
        q = questions[i] if isinstance(questions[i], dict) else {}
        q, _, qtype, _ = _q_meta(q, i)
        user_ans = _get_answer(i)

        if not _has_value(user_ans):
            return False

        if qtype == "mcq":
            correct_index = q.get("correct_index", q.get("answer_index"))
            try:
                correct_index = int(correct_index)
            except Exception:
                return False
            try:
                return int(user_ans) == correct_index
            except Exception:
                return False

        return short_answer_is_correct(str(user_ans or ""), q)

    def _recompute_score():
        score0 = 0
        for j in range(len(questions)):
            if _is_answered(j) and _compute_correct(j):
                score0 += 1
        return score0

    answered_count = sum(1 for j in range(len(questions)) if _is_answered(j))
    all_answered = (answered_count == len(questions))
    score = _recompute_score()

    # ---------------------------
    # REVIEW MODE
    # ---------------------------
    if bool(state.get("review", False)):
        st.divider()
        st.subheader("🔍 Review")
        st.caption(f"Answered: {answered_count}/{len(questions)}  •  Score: {score}/{len(questions)}")

        for j in range(len(questions)):
            qraw = questions[j] if isinstance(questions[j], dict) else {}
            q, qid, qtype, prompt = _q_meta(qraw, j)

            user_ans = _get_answer(j)
            is_ans = _is_answered(j)
            is_correct = is_ans and _compute_correct(j)

            user_txt = "—"
            correct_txt = "—"

            if qtype == "mcq":
                choices = q.get("choices", [])
                correct_index = q.get("correct_index", q.get("answer_index"))
                try:
                    correct_index = int(correct_index)
                except Exception:
                    correct_index = None

                try:
                    if _has_value(user_ans) and isinstance(choices, list) and 0 <= int(user_ans) < len(choices):
                        user_txt = f"{chr(65 + int(user_ans))}. {choices[int(user_ans)]}"
                except Exception:
                    user_txt = "—"

                if isinstance(choices, list) and correct_index is not None and 0 <= correct_index < len(choices):
                    correct_txt = f"{chr(65 + correct_index)}. {choices[correct_index]}"
            else:
                if _has_value(user_ans):
                    user_txt = str(user_ans)
                if q.get("answer", "") not in (None, ""):
                    correct_txt = str(q.get("answer", ""))

            status = "✅" if is_correct else ("❌" if is_ans else "⏳")
            with st.expander(f"Question {j+1} — {qid}  {status}", expanded=False):
                st.markdown(translate_latex(str(prompt)), unsafe_allow_html=True)
                st.markdown(f"**Your answer:** {translate_latex(str(user_txt))}", unsafe_allow_html=True)
                st.markdown(f"**Correct answer:** {translate_latex(str(correct_txt))}", unsafe_allow_html=True)

                explanation = q.get("explanation") or q.get("explain") or ""
                if explanation:
                    st.markdown("**Explanation**")
                    st.markdown(translate_latex(str(explanation)), unsafe_allow_html=True)

        st.divider()
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("⬅ Back to quiz", width="stretch", key=f"ws_{scope}_back_to_quiz"):
                state["review"] = False
                ss[state_key] = state
                st.rerun()
        with b2:
            if st.button("🔁 Restart quiz", width="stretch", key=f"ws_{scope}_restart_from_review"):
                _reset_all()
                st.rerun()
        return

    # ---------------------------
    # QUIZ MODE
    # ---------------------------
    i = _clamp_idx()
    qraw = questions[i] if isinstance(questions[i], dict) else {}
    q, qid, qtype, prompt = _q_meta(qraw, i)

    is_answered = _is_answered(i)
    saved_answer = _get_answer(i)

    # Header row
    h1, h2, h3 = st.columns([2, 1, 1], vertical_alignment="center")
    with h1:
        st.markdown(f"### Question {i+1} of {len(questions)} — {qid}")
        st.caption(f"Answered: {answered_count}/{len(questions)}")
    with h2:
        st.metric("Score", int(score))
    with h3:
        if st.button("↩ Reset", width="stretch", key=f"ws_{scope}_reset_{i}"):
            _reset_all()
            st.rerun()

    st.markdown(translate_latex(str(prompt)), unsafe_allow_html=True)

    # Widget keys (draft only) — safe to discard anytime
    w_key = f"ws_{scope}_w_{i}"

    user_ans = None

    if is_answered:
        # Show static answer (authoritative)
        if qtype == "mcq":
            choices = q.get("choices", [])
            display = "—"
            try:
                if _has_value(saved_answer) and isinstance(choices, list) and 0 <= int(saved_answer) < len(choices):
                    display = f"{chr(65 + int(saved_answer))}. {choices[int(saved_answer)]}"
            except Exception:
                display = "—"
            st.info(f"Your answer: {display}")
        else:
            st.info(f"Your answer: {str(saved_answer) if _has_value(saved_answer) else '—'}")
    else:
        # Not answered yet: input widget.
        # If they previously visited and chose something (draft), keep it by reading widget value.
        if qtype == "mcq":
            choices = q.get("choices", [])
            if not isinstance(choices, list) or not choices:
                st.warning("This MCQ has no choices.")
            else:
                # Optional: if you want draft persistence across navigation, seed index from ss[w_key]
                user_ans = st.radio(
                    "Choose one:",
                    options=list(range(len(choices))),
                    index=None,
                    format_func=lambda k: f"{chr(65+k)}. {choices[k]}",
                    key=w_key,
                )
        else:
            user_ans = st.text_input("Your answer:", key=w_key)

    # Actions
    a1, a2, a3, a4 = st.columns([1, 1, 1, 1])

    with a1:
        if st.button("✅ Submit", width="stretch", disabled=is_answered, key=f"ws_{scope}_submit_{i}"):

            if qtype == "mcq":
                # For MCQ, user_ans is the radio return; also stored in ss[w_key]
                val = ss.get(w_key, user_ans)
                if val is None:
                    st.warning("Please choose an option first.")
                    st.rerun()
            else:
                val = ss.get(w_key, user_ans)

            # Save submitted answer into authoritative state
            state["answers"][i] = val
            state["answered"][i] = True
            ss[state_key] = state
            st.rerun()

    with a2:
        if st.button("⟵ Prev", width="stretch", disabled=(i == 0), key=f"ws_{scope}_prev_{i}"):
            state["idx"] = max(0, i - 1)
            ss[state_key] = state
            st.rerun()

    with a3:
        if st.button(
            "Next ⟶",
            width="stretch",
            disabled=(i >= len(questions) - 1) or (not _is_answered(i)),
            key=f"ws_{scope}_next_{i}",
        ):
            state["idx"] = min(len(questions) - 1, i + 1)
            ss[state_key] = state
            st.rerun()

    with a4:
        if st.button("🔍 Review", width="stretch", disabled=(answered_count == 0), key=f"ws_{scope}_review_anytime"):
            state["review"] = True
            ss[state_key] = state
            st.rerun()

    # Feedback (only if answered)
    if is_answered:
        explanation = q.get("explanation") or q.get("explain") or ""

        if qtype == "mcq":
            choices = q.get("choices", [])
            correct_index = q.get("correct_index", q.get("answer_index"))
            try:
                correct_index = int(correct_index)
            except Exception:
                correct_index = None

            if correct_index is None or not choices:
                st.info("No answer key available for this question.")
            else:
                correct_text = (
                    f"{chr(65+correct_index)}. {choices[correct_index]}"
                    if 0 <= correct_index < len(choices)
                    else "(correct_index out of range)"
                )

                try:
                    ok = (saved_answer is not None) and (int(saved_answer) == correct_index)
                except Exception:
                    ok = False

                if ok:
                    st.success(f"✅ Correct — {correct_text}")
                else:
                    st.error(f"❌ Not quite. Correct answer: {correct_text}")

        else:
            saved_txt = str(saved_answer or "")
            if short_answer_is_correct(saved_txt, q):
                st.success("✅ Correct")
            else:
                st.error("❌ Incorrect")
                correct = q.get("answer", "")
                if correct:
                    st.markdown(
                        f"**Correct answer:** {translate_latex(str(correct))}",
                        unsafe_allow_html=True,
                    )

        if explanation:
            st.markdown("**Explanation**")
            st.markdown(translate_latex(str(explanation)), unsafe_allow_html=True)

    # Completion panel ONLY when ALL questions answered
    if all_answered:
        st.divider()
        st.success(f"✅ Finished! Score: {int(score)} / {len(questions)}")

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("🔁 Restart quiz", width="stretch", key=f"ws_{scope}_restart"):
                _reset_all()
                st.rerun()
        with c2:
            if st.button("🔍 Review questions", width="stretch", key=f"ws_{scope}_review_done"):
                state = ss[state_key]
                state["review"] = True
                ss[state_key] = state
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
