# numerace.py
import time
import streamlit as st
from streamlit import session_state as ss
from streamlit_autorefresh import st_autorefresh
from pathlib import Path
import sys
import textwrap

# Ensure project root is on sys.path so we can import from /shared
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.numeracy_dsl import load_game, pick_question_def, build_question
from shared.google_db import append_numerace_round

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

def timer_color(frac_left: float) -> str:
    """
    frac_left: 1.0 means full time left, 0.0 means time is up.
    Green -> Yellow -> Red.
    """
    if frac_left <= 0.20:
        return "#ef4444"  # red
    if frac_left <= 0.50:
        return "#f59e0b"  # yellow/amber
    return "#22c55e"      # green

def render_solid_track(correct: int, segments: int, q_seconds_left: float, q_seconds_total: float):
    segs = max(1, int(segments))
    correct_i = max(0, int(correct))

    prog = clamp01(correct_i / segs)
    pct = int(round(prog * 100))

    # keep racer visible inside ends (translateX(-50%) + edges)
    racer_pct = max(4, min(96, pct))

    finished = correct_i >= segs
    racer = "🏎️💨" if finished else "🏎️"
    extra = "nr-racer-finish" if finished else ""

    # force DOM change so finish animation can re-trigger
    burst = f"burst-{ss.get('nr_round_id', 0)}-{correct_i}"

    html = (
        f'<div class="nr-track-wrap">'
        f'  <div class="nr-track-flags">'
        f'    <div class="nr-flag nr-flag-start">🚩</div>'
        f'    <div class="nr-flag nr-flag-finish">🏁</div>'
        f'  </div>'
        f'  <div class="nr-track-stage">'
        f'    <div class="nr-track-line">'
        f'      <div class="nr-track-fill" style="width:{pct}%"></div>'
        f'    </div>'
        f'    <div class="nr-racer {extra}" data-burst="{burst}" style="left:{racer_pct}%">{racer}</div>'
        f'  </div>'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)

    if ss.nr_state in ("round_complete", "idle"):
        return

    secs = int(max(0.0, round(q_seconds_left)))
    cls = "nr-qcount nr-qcount-danger" if secs <= 3 else "nr-qcount"
    st.markdown(f"<div class='{cls}'>{secs}</div>", unsafe_allow_html=True)

def render_circular_timer(label: str, seconds_left: float, seconds_total: float, key: str):
    """
    Renders a circular progress ring using inline SVG.
    - seconds_left can be float
    - seconds_total must be > 0
    """
    total = max(1e-9, float(seconds_total))
    left = max(0.0, float(seconds_left))
    frac_left = clamp01(left / total)

    # ring geometry
    r = 18
    c = 2 * 3.141592653589793 * r
    prog = frac_left  # show "time remaining" as filled arc
    dash = prog * c
    gap = c - dash

    col = timer_color(frac_left)

    # show whole seconds
    txt = str(int(round(left)))

    st.markdown(
        f"""
        <div class="nr-ring" id="{key}">
          <svg viewBox="0 0 50 50" class="nr-ring-svg" aria-label="{label}">
            <circle class="nr-ring-bg" cx="25" cy="25" r="{r}"></circle>
            <circle class="nr-ring-fg" cx="25" cy="25" r="{r}"
              stroke="{col}"
              stroke-dasharray="{dash} {gap}">
            </circle>
            <text x="25" y="28" text-anchor="middle" class="nr-ring-text">{txt}</text>
          </svg>
          <div class="nr-ring-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

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

    ss.nr_q_time_frozen = question_time_left(time.time())

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
        /* ---- basic layout ---- */
        .nr-title { font-size: 34px; font-weight: 800; letter-spacing: 0.5px; }
        .nr-sub   { font-size: 14px; opacity: 0.8; margin-top: -6px; }

        .nr-card  { border: 1px solid rgba(15, 23, 42, 0.15); border-radius: 18px; padding: 18px; }
        .nr-qwrap { max-width: 900px; margin-left:auto; margin-right:auto; }

        .nr-prompt{ text-align:center; font-size: 22px; font-weight: 700; margin-bottom: 10px; }
        .nr-muted { opacity: 0.75; }

        /* ---- header rings (Round timer only) ---- */
        .nr-rings { display:flex; gap:14px; align-items:center; justify-content:flex-start; }
        .nr-ring  { display:flex; flex-direction:column; align-items:center; width:78px; }
        .nr-ring-svg { width:56px; height:56px; transform: rotate(-90deg); }
        .nr-ring-bg  { fill:none; stroke: rgba(15, 23, 42, 0.12); stroke-width: 6; }
        .nr-ring-fg  { fill:none; stroke-width: 6; stroke-linecap: round; }
        .nr-ring-text{ transform: rotate(90deg); font-size: 14px; font-weight: 800; fill: rgba(15, 23, 42, 0.85); }
        .nr-ring-label{ margin-top: 4px; font-size: 12px; font-weight: 700; opacity: 0.75; text-align:center; }

        /* ---- track: centered, shorter, flags ABOVE ---- */
        .nr-track-wrap{ max-width: 640px; margin: 10px auto 6px auto; }

        .nr-track-flags{
          display:flex;
          justify-content: space-between;
          align-items:center;
          padding: 0 2px 6px 2px;
        }

        .nr-flag{ font-size: 20px; line-height: 1; opacity: 0.85; }

        /* Stage holds both bar + racer (racer can stick out) */
        .nr-track-stage{
          position: relative;
          overflow: visible;   /* <-- allows car above the bar */
        }
        
        /* Bar itself keeps clipping for rounded fill */
        .nr-track-line{
          height: 10px;
          border-radius: 999px;
          background: rgba(15, 23, 42, 0.12);
          position: relative;
          overflow: hidden;    /* <-- clip ONLY the fill */
        }

        /* BLUE progress fill */
        .nr-track-fill{
          height: 100%;
          border-radius: 999px;
          background: linear-gradient(90deg,#3b82f6,#2563eb);
        }

        /* racer + flourish */
        .nr-racer{
          position:absolute;
          top:-16px;
          transform: translateX(-50%) scaleX(-1);
          font-size: 36px;
          transition: left 0.45s ease;
          will-change: left, transform;
          z-index: 5;
        }

        .nr-racer-finish{ animation: nr-finish-burst 0.6s ease-out; }

        @keyframes nr-finish-burst{
          0%   { transform: translateX(-50%) scaleX(-1) scale(1); }
          40%  { transform: translateX(-50%) scaleX(-1) scale(1.35) rotate(-8deg); }
          70%  { transform: translateX(-50%) scaleX(-1) scale(1.25) rotate(6deg); }
          100% { transform: translateX(-50%) scaleX(-1) scale(1.1); }
        }

        /* ---- big circular question countdown under the bar ---- */
        .nr-qcount{
          margin: 12px auto 0 auto;
          width: 64px;
          height: 64px;
          border-radius: 50%;
          border: 3px solid rgba(15,23,42,0.25);
          background: rgba(255,255,255,0.6);
          display:flex;
          align-items:center;
          justify-content:center;
          font-size: 24px;
          font-weight: 800;
          box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        }

        .nr-qcount-danger{ border-color: #ef4444; color:#ef4444; }

        /* ---- per-choice feedback cards (used under choices) ---- */
        .nr-fb {
          margin-top: 10px;
          padding: 12px 14px;
          border-radius: 14px;
          border: 1px solid rgba(15, 23, 42, 0.12);
          background: rgba(255,255,255,0.55);
          font-size: 18px;
          line-height: 1.25;
        }
        .nr-fb small { font-size: 13px; opacity: 0.8; }
        .nr-fb .nr-fb-val { font-size: 22px; font-weight: 800; }

        .nr-fb-ok  { border-color: rgba(34,197,94,0.45);  background: rgba(34,197,94,0.10); }
        .nr-fb-bad { border-color: rgba(239,68,68,0.45); background: rgba(239,68,68,0.08); }
        .nr-fb-neu { border-color: rgba(148,163,184,0.35); background: rgba(148,163,184,0.10); }

        /* button rounding (optional) */
        div[data-testid="stButton"] > button { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header: (left) circular timers | (mid) title | (right) control button
    hL, hM, hR = st.columns([1.3, 2.7, 1.2], vertical_alignment="center")

    with hL:
        # compute timer values (idle shows full defaults)
        rt_left = max(0.0, round_time_left(now))

        # show question ring only when round is running; otherwise still show full
        st.markdown('<div class="nr-rings">', unsafe_allow_html=True)
        render_circular_timer("Round", rt_left, ss.nr_round_seconds, key="nr_ring_round")
        st.markdown("</div>", unsafe_allow_html=True)

    with hM:
        st.markdown("<div class='nr-title'>🏁 NumeRace</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='nr-sub'>Round #{ss.nr_round_id} • "
            f"Questions: {ss.nr_correct_in_round}/{ss.nr_segments}</div>",
            unsafe_allow_html=True
        )

    with hR:
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
            if st.button("🛑 Cancel", width="stretch"):
                ss.nr_state = "round_complete"
                ss.nr_feedback = None
                ss.nr_feedback_started_at = None
                ss.nr_round_pause_started_at = None
                st.rerun()

    # ------------------------------------------------------------
    # Track row with Cancel on the left
    # ------------------------------------------------------------
    track_mid = st.columns([1], vertical_alignment="center")[0]

    with track_mid:

        if ss.nr_state == "feedback":
            qt_left = ss.nr_q_time_frozen
        else:
            qt_left = max(0.0, question_time_left(now))

        render_solid_track(
            ss.nr_correct_in_round,
            ss.nr_segments,
            qt_left,
            ss.nr_question_seconds
        )

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
        return
    # Round complete summary (Total / Correct / Incorrect / Missed)
    elif ss.nr_state == "round_complete":
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

        return

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

