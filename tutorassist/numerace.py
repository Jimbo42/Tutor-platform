# numerace.py
import time
import streamlit as st
from streamlit import session_state as ss
from streamlit_autorefresh import st_autorefresh
from pathlib import Path
import sys

# Ensure project root is on sys.path so we can import from /shared
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.numeracy_dsl import load_game, pick_question_def, build_question
from shared.google_sheets_db import append_numerace_round

# ------------------------------------------------------------
# Config (tune these later / per difficulty)
# ------------------------------------------------------------
SEGMENTS_PER_ROUND = 8

ROUND_SECONDS_DEFAULT = 90          # total time to finish 8 correct answers
QUESTION_SECONDS_DEFAULT = 10        # time per question
FEEDBACK_SECONDS_DEFAULT = 3.5      # paused feedback display time (excluded from round timer)

AUTOREFRESH_ANSWER_MS = 1000   # slow: avoids click-eating
AUTOREFRESH_FEEDBACK_MS = 250  # fast: snappy auto-advance

# ------------------------------------------------------------
# Helpers: session-state init
# ------------------------------------------------------------
def ss_init():
    if "nr_initialized" in ss:
        return

    ss.nr_initialized = True

    ss.nr_segments = SEGMENTS_PER_ROUND
    ss.nr_round_seconds = ROUND_SECONDS_DEFAULT
    ss.nr_question_seconds = QUESTION_SECONDS_DEFAULT
    ss.nr_feedback_seconds = FEEDBACK_SECONDS_DEFAULT

    ss.nr_round_id = 1
    ss.nr_correct_in_round = 0

    # start idle until user clicks "Start round"
    ss.nr_state = "idle"   # idle | answering | feedback | round_complete

    # timers (not running yet)
    ss.nr_round_started_at = None
    ss.nr_round_paused_accum = 0.0
    ss.nr_round_pause_started_at = None

    ss.nr_q_started_at = None

    # current question payload (none until start)
    ss.nr_q = None
    ss.nr_last_choice = None
    ss.nr_feedback = None
    ss.nr_feedback_started_at = None
    ss.nr_no_refresh_until = 0.0

    # optional: stats
    ss.nr_attempts_total = 0
    ss.nr_correct_total = 0

    # Round breakdown
    ss.nr_incorrect_in_round = 0
    ss.nr_missed_in_round = 0

    # Totals breakdown
    ss.nr_incorrect_total = 0
    ss.nr_missed_total = 0

    # DSL game config
    ss.nr_game_path = str(PROJECT_ROOT / "shared" / "numeracy_game.json")
    ss.nr_game = load_game(ss.nr_game_path)

    # selection tracking within a round
    ss.nr_q_history = []      # list of question IDs used recently
    ss.nr_q_used_counts = {}  # qid -> count used this round

    # pull defaults from JSON rules (still overridable by sliders if you want)
    rules = ss.nr_game.get("rules", {})
    ss.nr_segments = int(rules.get("segments_per_round", ss.nr_segments))
    ss.nr_round_seconds = int(rules.get("round_seconds", ss.nr_round_seconds))
    ss.nr_question_seconds = int(rules.get("question_seconds", ss.nr_question_seconds))
    ss.nr_feedback_seconds = int(rules.get("feedback_seconds", ss.nr_feedback_seconds))

    ss.nr_logged_round_id = None
    ss.nr_resp_times = []

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def suppress_autorefresh(seconds: float = 0.5):
    ss.nr_no_refresh_until = time.time() + seconds

def start_round():
    """Start (or restart) the timers and load the first question."""
    now = time.time()

    ss.nr_state = "answering"
    ss.nr_round_started_at = now
    ss.nr_round_paused_accum = 0.0
    ss.nr_round_pause_started_at = None

    ss.nr_q_started_at = now
    ss.nr_q = make_question()
    suppress_autorefresh()

    ss.nr_last_choice = None
    ss.nr_feedback = None
    ss.nr_feedback_started_at = None

    ss.nr_incorrect_in_round = 0
    ss.nr_missed_in_round = 0

    ss.nr_q_history = []
    ss.nr_q_used_counts = {}

    ss.nr_logged_round_id = None
    ss.nr_resp_times = []

def reset_round():
    ss.nr_correct_in_round = 0

    # back to idle — user must click Start
    ss.nr_state = "idle"

    ss.nr_round_started_at = None
    ss.nr_round_paused_accum = 0.0
    ss.nr_round_pause_started_at = None

    ss.nr_q_started_at = None
    ss.nr_q = None

    ss.nr_last_choice = None
    ss.nr_feedback = None
    ss.nr_feedback_started_at = None

    ss.nr_incorrect_in_round = 0
    ss.nr_missed_in_round = 0

    ss.nr_q_history = []
    ss.nr_q_used_counts = {}

    ss.nr_logged_round_id = None
    ss.nr_resp_times = []

def start_feedback_pause():
    """Enter feedback state; exclude this time from the round timer."""
    now = time.time()

    ss.nr_state = "feedback"
    ss.nr_feedback_started_at = now

    # start excluding from round timer
    ss.nr_round_pause_started_at = now

def end_feedback_pause_and_advance():
    """Leave feedback state, add excluded time to accumulator, and advance to next question or end round."""
    now = time.time()

    # accumulate excluded time
    if ss.nr_round_pause_started_at is not None:
        ss.nr_round_paused_accum += (now - ss.nr_round_pause_started_at)
        ss.nr_round_pause_started_at = None

    # if round complete, stop; else next question
    if ss.nr_correct_in_round >= ss.nr_segments:
        ss.nr_state = "round_complete"

        # ✅ log on transition into round_complete (idempotent)
        total_q = ss.nr_correct_in_round + ss.nr_incorrect_in_round + ss.nr_missed_in_round
        try:
            log_round_once(total_q)
        except Exception as e:
            st.warning(f"Could not save to Google Sheets: {e}")

        return

    ss.nr_state = "answering"
    ss.nr_q_started_at = now
    ss.nr_q = make_question()
    suppress_autorefresh()

    ss.nr_last_choice = None
    ss.nr_feedback = None
    ss.nr_feedback_started_at = None


# ------------------------------------------------------------
# Timers
# ------------------------------------------------------------
def round_time_elapsed_active(now: float) -> float:
    """Round timer counts only while 'answering' (feedback time is excluded)."""
    if ss.nr_round_started_at is None:
        return 0.0  # idle / not started yet

    paused_extra = ss.nr_round_paused_accum

    # If currently in feedback, also exclude the currently-running pause segment
    if ss.nr_round_pause_started_at is not None:
        paused_extra += (now - ss.nr_round_pause_started_at)

    return (now - ss.nr_round_started_at) - paused_extra


def round_time_left(now: float) -> float:
    # If idle/not started, show full time remaining
    if ss.nr_round_started_at is None:
        return float(ss.nr_round_seconds)
    return ss.nr_round_seconds - round_time_elapsed_active(now)


def question_time_left(now: float) -> float:
    # If idle/not started, show full time remaining
    if ss.nr_q_started_at is None:
        return float(ss.nr_question_seconds)
    return ss.nr_question_seconds - (now - ss.nr_q_started_at)

# ------------------------------------------------------------
# Question generation
# ------------------------------------------------------------
def make_question():
    qdef = pick_question_def(ss.nr_game, ss.nr_q_history, ss.nr_q_used_counts)
    built = build_question(ss.nr_game, qdef)

    # update tracking
    ss.nr_q_history.append(built.qid)
    ss.nr_q_used_counts[built.qid] = ss.nr_q_used_counts.get(built.qid, 0) + 1

    return {
        "prompt": built.prompt,
        "choices": built.choices,               # [{"label":..., "value":...}, ...]
        "correct_index": built.correct_index,
        "explain": built.explain or "",
        "qid": built.qid
    }

# ------------------------------------------------------------
# Game logic
# ------------------------------------------------------------
def handle_timeout_if_needed(now: float):

    if ss.nr_state != "answering":
        return

    # Round timeout ends the round
    if round_time_left(now) <= 0:
        ss.nr_state = "round_complete"
        ss.nr_feedback = None
        ss.nr_feedback_started_at = None
        ss.nr_round_pause_started_at = None

        # ✅ log on transition into round_complete (idempotent)
        total_q = ss.nr_correct_in_round + ss.nr_incorrect_in_round + ss.nr_missed_in_round
        try:
            log_round_once(total_q)
        except Exception as e:
            st.warning(f"Could not save to Google Sheets: {e}")

        return

    # Question timeout counts as a MISS (not incorrect), shows feedback, then advances
    if question_time_left(now) <= 0:
        ss.nr_attempts_total += 1
        ss.nr_last_choice = None

        ss.nr_missed_in_round += 1
        ss.nr_missed_total += 1

        # track response time for a missed question (full time window)
        ss.nr_resp_times.append(float(ss.nr_question_seconds))

        ss.nr_feedback = {
            "kind": "miss",
            "is_correct": False,
            "msg": "⏱️ Missed (no answer)",
            "correct_label": ss.nr_q["choices"][ss.nr_q["correct_index"]]["label"],
            "explain": ss.nr_q["explain"],
        }
        start_feedback_pause()
        return

def submit_answer(choice_index: int):
    if ss.nr_state != "answering":
        return

    ss.nr_attempts_total += 1
    ss.nr_last_choice = choice_index
    # track response time for this question
    if ss.nr_q_started_at is not None:
        ss.nr_resp_times.append(time.time() - ss.nr_q_started_at)

    correct = (choice_index == ss.nr_q["correct_index"])
    if correct:
        ss.nr_correct_total += 1
        ss.nr_correct_in_round += 1
    else:
        ss.nr_incorrect_in_round += 1
        ss.nr_incorrect_total += 1

    correct_label = ss.nr_q["choices"][ss.nr_q["correct_index"]]["label"]
    ss.nr_feedback = {
        "kind": "answer",  # NEW
        "is_correct": correct,
        "msg": "✅ Correct!" if correct else "❌ Incorrect",
        "correct_label": correct_label,
        "explain": ss.nr_q["explain"],
    }
    start_feedback_pause()

def tick_feedback(now: float):
    if ss.nr_state != "feedback":
        return

    if (now - ss.nr_feedback_started_at) >= ss.nr_feedback_seconds:
        end_feedback_pause_and_advance()

def render_linear_track(completed: int, total: int):
    """
    Draw 8 (or N) horizontal segments. Completed segments are filled.
    """
    segs = []
    for i in range(total):
        segs.append("nr-seg-on" if i < completed else "nr-seg-off")

    st.markdown(
        """
        <style>
        .nr-track {
            display: flex;
            gap: 10px;
            align-items: center;
            justify-content: center;
            padding: 12px 4px 18px 4px;
        }
        .nr-seg {
            height: 14px;
            width: 90px;
            border-radius: 999px;
            border: 1px solid rgba(15, 23, 42, 0.18);
            box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        }
        .nr-seg-on  { background: #22c55e; opacity: 1.0; }
        .nr-seg-off { background: #94a3b8; opacity: 0.25; }
        </style>
        """,
        unsafe_allow_html=True
    )

    html = "<div class='nr-track'>" + "".join([f"<div class='nr-seg {c}'></div>" for c in segs]) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

def log_round_once(total_q: int):
    """
    Log the current round to Google Sheets exactly once per round_id.
    Safe across Streamlit reruns.
    """
    # ✅ already logged this specific round
    if ss.get("nr_logged_round_id") == ss.nr_round_id:
        return

    ss.nr_logged_round_id = ss.nr_round_id

    quiz_id = f"nr_round_{ss.nr_round_id}"
    username = ss.get("username", "unknown")
    total_questions = ss.nr_correct_in_round + ss.nr_incorrect_in_round + ss.nr_missed_in_round
    now = time.time()
    round_time = float(round_time_elapsed_active(now))

    # average_response_time: mean of per-question response times
    if ss.nr_resp_times:
        average_response_time = float(sum(ss.nr_resp_times) / len(ss.nr_resp_times))
    else:
        average_response_time = 0.0

    append_numerace_round(
        username=username,
        total_questions=int(total_questions),
        incorrect=int(ss.nr_incorrect_in_round),
        missed=int(ss.nr_missed_in_round),
        attempts_total=int(ss.nr_attempts_total),
        round_time=round_time,
        average_response_time=average_response_time,
    )

    # time.sleep(0.25)

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
def numerace_app():

    st.set_page_config(page_title="NumeRace", layout="wide")
    ss_init()

    now = time.time()

    # Only tick/refresh while actively playing
    if ss.nr_state in ("answering", "feedback"):

        # Always schedule refresh (keeps timers/feedback moving)
        if ss.nr_state == "answering":
            st_autorefresh(interval=AUTOREFRESH_ANSWER_MS, key="nr_tick_answer")
        else:
            st_autorefresh(interval=AUTOREFRESH_FEEDBACK_MS, key="nr_tick_feedback")

        # During the short suppression window, we do NOT run logic that can
        # interfere with clicks / immediate transitions.
        if time.time() >= ss.nr_no_refresh_until:
            if ss.nr_state == "answering":
                handle_timeout_if_needed(now)
            else:
                tick_feedback(now)

    # ------------------------------------------------------------
    # Top header + styles
    # ------------------------------------------------------------
    st.markdown(
        """
        <style>
        .nr-title { font-size: 34px; font-weight: 800; letter-spacing: 0.5px; }
        .nr-sub { font-size: 14px; opacity: 0.8; margin-top: -6px; }

        .nr-card { border: 1px solid rgba(15, 23, 42, 0.15); border-radius: 18px; padding: 18px; }
        .nr-qwrap{ max-width: 900px; margin-left:auto; margin-right:auto; }

        .nr-prompt{ text-align:center; font-size: 22px; font-weight: 700; margin-bottom: 10px; }
        .nr-timer { font-size: 16px; font-weight: 700; }
        .nr-muted { opacity: 0.75; }

        /* Make header button a bit tighter */
        div[data-testid="stButton"] > button { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header: Title | Round timer | Start/Next button
    h1, h2, h3 = st.columns([2.2, 2.4, 1.2], vertical_alignment="center")

    with h1:
        st.markdown("<div class='nr-title'>🏁 NumeRace</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='nr-sub'>Round #{ss.nr_round_id} • "
            f"Questions: {ss.nr_correct_in_round}/{ss.nr_segments}</div>",
            unsafe_allow_html=True
        )

    # Round progress (no seconds text)
    rt = max(0.0, round_time_left(now))
    with h2:
        st.markdown("<div class='nr-timer'>⏱️ Round</div>", unsafe_allow_html=True)
        round_frac = clamp01((ss.nr_round_seconds - rt) / ss.nr_round_seconds)
        st.progress(round_frac)

    # Header control button (Start / Next)
    with h3:
        if ss.nr_state == "idle":
            if st.button("🏁 Start round", width="stretch"):
                start_round()
                st.rerun()

        elif ss.nr_state == "round_complete":
            if st.button("➡️ Next round", width="stretch"):
                ss.nr_round_id += 1
                reset_round()
                st.rerun()

        else:
            # answering / feedback
            if st.button("🛑 Cancel", width="stretch"):
                ss.nr_state = "round_complete"
                ss.nr_feedback = None
                ss.nr_feedback_started_at = None
                ss.nr_round_pause_started_at = None
                st.rerun()

    st.divider()

    # ------------------------------------------------------------
    # Track row with Cancel on the left
    # ------------------------------------------------------------
    track_mid = st.columns([1], vertical_alignment="center")[0]

    with track_mid:
        render_linear_track(ss.nr_correct_in_round, ss.nr_segments)

    # ------------------------------------------------------------
    # Center question container
    # ------------------------------------------------------------
    st.markdown("<div class='nr-card'><div class='nr-qwrap'>", unsafe_allow_html=True)

    # Idle screen (no start button here—it's in the header)
    if ss.nr_state == "idle":
        st.markdown("<div class='nr-prompt'>Ready to race?</div>", unsafe_allow_html=True)
        st.markdown("<div class='nr-muted' style='text-align:center;'>Press <b>Start round</b> in the header.</div>",
                    unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

        # Optional settings expander still visible in idle
        with st.expander("⚙️ NumeRace settings (temporary)", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                ss.nr_round_seconds = st.slider("Round seconds", 20, 180, ss.nr_round_seconds)
            with c2:
                ss.nr_question_seconds = st.slider("Question seconds", 3, 20, ss.nr_question_seconds)
            with c3:
                ss.nr_feedback_seconds = st.slider("Feedback seconds", 1, 6, int(ss.nr_feedback_seconds))
            st.caption("These will become difficulty presets later.")
        return

    # Round complete summary (Total / Correct / Incorrect / Missed)
    if ss.nr_state == "round_complete":
        st.markdown("<div class='nr-prompt'>🎉 Round complete!</div>", unsafe_allow_html=True)

        total_q = ss.nr_correct_in_round + ss.nr_incorrect_in_round + ss.nr_missed_in_round

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total questions", total_q)
        with c2:
            st.metric("Correct", ss.nr_correct_in_round)
        with c3:
            st.metric("Incorrect", ss.nr_incorrect_in_round)
        with c4:
            st.metric("Missed", ss.nr_missed_in_round)

        st.markdown("<div class='nr-muted' style='text-align:center;'>Press <b>Next round</b> in the header when ready.</div>",
                    unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        with st.expander("⚙️ NumeRace settings (temporary)", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                ss.nr_round_seconds = st.slider("Round seconds", 20, 180, ss.nr_round_seconds)
            with c2:
                ss.nr_question_seconds = st.slider("Question seconds", 3, 20, ss.nr_question_seconds)
            with c3:
                ss.nr_feedback_seconds = st.slider("Feedback seconds", 1, 6, int(ss.nr_feedback_seconds))
            st.caption("These will become difficulty presets later.")
        return

    # Question timer (progress only, no seconds text)
    if ss.nr_state == "answering":
        qt = max(0.0, question_time_left(now))
        q_frac = clamp01((ss.nr_question_seconds - qt) / ss.nr_question_seconds)
        st.markdown("<div class='nr-timer'>⚡ Question</div>", unsafe_allow_html=True)
        st.progress(q_frac)
    else:
        st.write("")

    # Normal question
    q = ss.nr_q
    st.markdown(f"<div class='nr-prompt'>{q['prompt']}</div>", unsafe_allow_html=True)

    # Choices + per-column feedback placeholders
    choice_cols = st.columns(len(q["choices"]))
    under = []

    for i, ch in enumerate(q["choices"]):
        with choice_cols[i]:
            disabled = (ss.nr_state != "answering")
            if st.button(
                ch["label"],
                key=f"nr_choice_{ss.nr_round_id}_{q['qid']}_{i}",
                width="stretch",
                disabled=disabled
            ):
                submit_answer(i)
                st.rerun()

            under.append(st.empty())

    # Per-column feedback (bigger + clearer)
    if ss.nr_state == "feedback" and ss.nr_feedback:
        # Inject once per run (safe)
        st.markdown(
            """
            <style>
            .nr-fb {
                margin-top: 10px;
                padding: 12px 14px;
                border-radius: 14px;
                border: 1px solid rgba(15, 23, 42, 0.12);
                background: rgba(255,255,255,0.55);
                font-size: 18px;
                line-height: 1.25;
            }
            .nr-fb small {
                font-size: 13px;
                opacity: 0.8;
            }
            .nr-fb .nr-fb-val {
                font-size: 22px;
                font-weight: 800;
            }
            .nr-fb-ok { border-color: rgba(34,197,94,0.45); background: rgba(34,197,94,0.10); }
            .nr-fb-bad { border-color: rgba(239,68,68,0.45); background: rgba(239,68,68,0.08); }
            .nr-fb-neu { border-color: rgba(148,163,184,0.35); background: rgba(148,163,184,0.10); }
            </style>
            """,
            unsafe_allow_html=True,
        )

        correct_i = q["correct_index"]
        fb = ss.nr_feedback

        for i, ch in enumerate(q["choices"]):
            # show label and big value
            label = ch["label"]
            val = ch["value"]

            if i == correct_i:
                cls = "nr-fb nr-fb-ok"
                head = "✅ Correct"
            elif ss.nr_last_choice == i:
                cls = "nr-fb nr-fb-bad"
                head = "❌ Your choice"
            else:
                cls = "nr-fb nr-fb-neu"
                head = "⬜"

            under[i].markdown(
                f"""
                <div class="{cls}">
                  <div><b>{head}</b></div>
                  <small>{label}</small><br/>
                  <span class="nr-fb-val">{val}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        remaining = max(0.0, ss.nr_feedback_seconds - (now - ss.nr_feedback_started_at))
        st.markdown(
            f"<div style='text-align:center; font-size:16px; margin-top:12px;'><b>{fb.get('msg', '')}</b> • continuing in {remaining:0.1f}s…</div>",
            unsafe_allow_html=True,
        )

    # Footer controls (optional)
    with st.expander("⚙️ NumeRace settings (temporary)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            ss.nr_round_seconds = st.slider("Round seconds", 20, 180, ss.nr_round_seconds)
        with c2:
            ss.nr_question_seconds = st.slider("Question seconds", 3, 20, ss.nr_question_seconds)
        with c3:
            ss.nr_feedback_seconds = st.slider("Feedback seconds", 1, 6, int(ss.nr_feedback_seconds))

        st.caption("These will become difficulty presets later.")
