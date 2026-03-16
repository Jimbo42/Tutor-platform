# numerace.py
import time
import streamlit as st
from streamlit import session_state as ss
from streamlit_autorefresh import st_autorefresh
from pathlib import Path
import sys
import uuid
import textwrap

# Ensure project root is on sys.path so we can import from /shared
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.numeracy_dsl import load_game, pick_question_def, build_question
from shared.google_db import append_numerace_round, append_numerace_attempt_rows, append_row_by_header

# ------------------------------------------------------------
# Config (tune these later / per difficulty)
# ------------------------------------------------------------
SEGMENTS_PER_ROUND = 8

ROUND_SECONDS_DEFAULT = 90          # total time to finish 8 correct answers
QUESTION_SECONDS_DEFAULT = 10        # time per question
FEEDBACK_SECONDS_DEFAULT = 3.5      # paused feedback display time (excluded from round timer)
DIFFICULTY_LEVELS = ["Starter", "Intermediate", "Challenging"]

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
    ss.nr_state = "idle"   # idle | answering | feedback | saving_round | round_complete

    # timers (not running yet)
    ss.nr_round_started_at = None
    ss.nr_round_paused_accum = 0.0
    ss.nr_round_pause_started_at = None

    ss.nr_q_started_at = None
    ss.nr_q_time_frozen = float(ss.nr_question_seconds)

    # current question payload (none until start)
    ss.nr_q = None
    ss.nr_last_choice = None
    ss.nr_feedback = None
    ss.nr_feedback_started_at = None
    ss.nr_no_refresh_until = 0.0

    # optional: stats
    ss.nr_attempts_total = 0
    ss.nr_correct_total = 0
    ss.nr_difficulty = "Starter"
    ss.nr_start_error = None
    ss.nr_round_start_difficulty = ss.nr_difficulty
    ss.nr_round_end_reason = ""

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

    # pull defaults from JSON rules
    rules = ss.nr_game.get("rules", {})
    ss.nr_segments = int(rules.get("segments_per_round", ss.nr_segments))
    ss.nr_round_seconds = int(rules.get("round_seconds", ss.nr_round_seconds))
    ss.nr_question_seconds = int(rules.get("question_seconds", ss.nr_question_seconds))
    ss.nr_feedback_seconds = int(rules.get("feedback_seconds", ss.nr_feedback_seconds))

    ss.nr_logged_round_id = None
    ss.nr_resp_times = []
    ss.nr_question_seq = 0
    ss.nr_current_attempts = 0
    ss.nr_attempt_buffer = []
    ss.nr_last_attempt_flush_error = ""

    # NEW: stable ids for idempotent logging
    ss.nr_session_id = uuid.uuid4().hex[:12]
    ss.nr_round_key = ""
    ss.nr_last_round_score = None
    ss.nr_score_toast_shown_for_round = None
    ss.nr_round_summary = None

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

    # keep racer visible inside ends
    racer_pct = max(4, min(96, pct))

    finished = correct_i >= segs
    extra = "nr-racer-finish" if finished else ""

    # place smoke just behind the car when finished
    smoke_pct = max(3.0, racer_pct - 1.6)

    burst = f"burst-{ss.get('nr_round_id', 0)}-{correct_i}"

    html = (
        f'  <div class="nr-track-flags">'
        f'    <div class="nr-flag nr-flag-start">🚩</div>'
        f'    <div class="nr-flag nr-flag-finish">🏁</div>'
        f'  </div>'
        f'  <div class="nr-track-stage">'
        f'    <div class="nr-track-line">'
        f'      <div class="nr-track-fill" style="width:{pct}%"></div>'
        f'    </div>'
        + (
            f'    <div class="nr-smoke" style="left:{smoke_pct}%">💨</div>'
            if finished else ""
        )
        + f'    <div class="nr-racer {extra}" data-burst="{burst}" style="left:{racer_pct}%">🏎️</div>'
        f'  </div>'
    )

    st.markdown(html, unsafe_allow_html=True)

    if ss.nr_state in ("round_complete", "saving_round", "idle"):
        return

    secs = int(max(0.0, round(q_seconds_left)))
    cls = "nr-qcount nr-qcount-danger" if secs <= 3 else "nr-qcount"
    st.markdown(f"<div class='{cls}'>{secs}</div>", unsafe_allow_html=True)

def render_circular_timer(label: str, seconds_left: float, seconds_total: float, key: str):
    total = max(1e-9, float(seconds_total))
    left = max(0.0, float(seconds_left))
    frac_left = clamp01(left / total)

    r = 18
    c = 2 * 3.141592653589793 * r
    dash = frac_left * c
    gap = c - dash

    col = timer_color(frac_left)
    txt = str(int(round(left)))
    extra_cls = " nr-ring-danger" if frac_left <= 0.20 else ""

    html = (
        f'<div class="nr-ring{extra_cls}" id="{key}">'
        f'<svg viewBox="0 0 50 50" class="nr-ring-svg">'
        f'<circle class="nr-ring-bg" cx="25" cy="25" r="{r}"></circle>'
        f'<circle class="nr-ring-fg" cx="25" cy="25" r="{r}" '
        f'stroke="{col}" stroke-dasharray="{dash} {gap}"></circle>'
        f'<text x="25" y="26" text-anchor="middle" dominant-baseline="middle" class="nr-ring-text">{txt}</text>'
        f'</svg></div>'
    )

    st.markdown(html, unsafe_allow_html=True)

def suppress_autorefresh(seconds: float = 0.5):
    ss.nr_no_refresh_until = time.time() + seconds

def start_round():
    """Start (or restart) the timers and load the first question."""
    now = time.time()
    ss.nr_start_error = None
    ss.nr_round_start_difficulty = ss.get("nr_difficulty", "Starter")
    ss.nr_round_end_reason = ""

    username = str(ss.get("username", "unknown")).strip() or "unknown"
    ss.nr_round_key = f"{username}|{ss.nr_session_id}|round_{ss.nr_round_id}"

    ss.nr_state = "answering"
    ss.nr_round_started_at = now
    ss.nr_round_paused_accum = 0.0
    ss.nr_round_pause_started_at = None
    ss.nr_q_time_frozen = float(ss.nr_question_seconds)

    ss.nr_q_started_at = now

    ss.nr_incorrect_in_round = 0
    ss.nr_missed_in_round = 0

    ss.nr_q_history = []
    ss.nr_q_used_counts = {}

    ss.nr_logged_round_id = None
    ss.nr_resp_times = []
    ss.nr_question_seq = 0
    ss.nr_current_attempts = 0
    ss.nr_attempt_buffer = []
    ss.nr_last_attempt_flush_error = ""

    try:
        ss.nr_q = make_question()
    except Exception as e:
        ss.nr_state = "idle"
        ss.nr_round_started_at = None
        ss.nr_q_started_at = None
        ss.nr_q = None
        ss.nr_start_error = str(e)
        return

    suppress_autorefresh()
    ss.nr_last_choice = None
    ss.nr_feedback = None
    ss.nr_feedback_started_at = None

def reset_round():
    ss.nr_correct_in_round = 0

    # back to idle - user must click Start
    ss.nr_state = "idle"

    ss.nr_round_started_at = None
    ss.nr_round_paused_accum = 0.0
    ss.nr_round_pause_started_at = None
    ss.nr_q_time_frozen = float(ss.nr_question_seconds)

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
    ss.nr_question_seq = 0
    ss.nr_current_attempts = 0
    ss.nr_attempt_buffer = []
    ss.nr_round_end_reason = ""
    ss.nr_last_attempt_flush_error = ""
    ss.nr_round_key = ""
    ss.nr_last_round_score = None
    ss.nr_score_toast_shown_for_round = None
    ss.nr_round_summary = None

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
        ss.nr_round_end_reason = "completed"
        prepare_round_summary()
        ss.nr_state = "saving_round"
        ss.nr_feedback = None
        ss.nr_feedback_started_at = None
        return

    ss.nr_state = "answering"
    ss.nr_q_started_at = now
    ss.nr_q = make_question()
    ss.nr_q_time_frozen = float(ss.nr_question_seconds)

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
    qdef = pick_question_def(
        ss.nr_game,
        ss.nr_q_history,
        ss.nr_q_used_counts,
        difficulty=ss.get("nr_difficulty", "Starter"),
    )
    built = build_question(ss.nr_game, qdef)

    # update tracking
    ss.nr_q_history.append(built.qid)
    ss.nr_q_used_counts[built.qid] = ss.nr_q_used_counts.get(built.qid, 0) + 1
    ss.nr_question_seq += 1
    ss.nr_current_attempts = 0

    classification = qdef.get("classification", {}) if isinstance(qdef, dict) else {}
    reporting = qdef.get("reporting", {}) if isinstance(qdef, dict) else {}
    tags = classification.get("tags", [])

    # Try to preserve generated values for debugging/reporting
    generated_values = {}
    for attr in ("vars", "values", "context", "derived", "env"):
        if hasattr(built, attr):
            try:
                val = getattr(built, attr)
                if isinstance(val, dict):
                    generated_values[attr] = val
            except Exception:
                pass

    return {
        "prompt": built.prompt,
        "choices": built.choices,               # [{"label":..., "value":...}, ...]
        "correct_index": built.correct_index,
        "explain": built.explain or "",
        "qid": built.qid,
        "title": qdef.get("title", built.qid),
        "difficulty": qdef.get("difficulty", ss.get("nr_difficulty", "Starter")),
        "classification": {
            "domain": classification.get("domain", ""),
            "skill": classification.get("skill", ""),
            "subskill": classification.get("subskill", ""),
            "tags": tags if isinstance(tags, list) else []
        },
        "reporting": {
            "track_accuracy": reporting.get("track_accuracy", True),
            "track_response_time": reporting.get("track_response_time", True),
            "track_attempts": reporting.get("track_attempts", True),
            "mastery_group": reporting.get("mastery_group", "")
        },
        "generated_values": generated_values
    }

# ------------------------------------------------------------
# Game logic
# ------------------------------------------------------------
def handle_timeout_if_needed(now: float):

    if ss.nr_state != "answering":
        return

    # Round timeout ends the round
    if round_time_left(now) <= 0:
        ss.nr_round_end_reason = "timed_out"
        ss.nr_feedback = None
        ss.nr_feedback_started_at = None
        ss.nr_round_pause_started_at = None
        prepare_round_summary()
        ss.nr_state = "saving_round"
        return

    # Question timeout counts as a miss, then advances after feedback.
    if question_time_left(now) <= 0:
        ss.nr_attempts_total += 1
        ss.nr_current_attempts += 1
        ss.nr_last_choice = None

        ss.nr_missed_in_round += 1
        ss.nr_missed_total += 1

        resp_time = float(ss.nr_question_seconds)
        ss.nr_resp_times.append(resp_time)

        # Transition immediately to prevent duplicate timeout handling.
        ss.nr_feedback = {
            "kind": "miss",
            "is_correct": False,
            "msg": "Missed (no answer)",
            "correct_label": ss.nr_q["choices"][ss.nr_q["correct_index"]]["label"],
            "explain": ss.nr_q["explain"],
        }
        start_feedback_pause()
        suppress_autorefresh()

        try:
            log_question_attempt(
                correct=False,
                missed=True,
                response_time=resp_time,
                selected_answer="",
            )
        except Exception as e:
            st.warning(f"Could not save question attempt: {e}")
        return
def submit_answer(choice_index: int):
    if ss.nr_state != "answering":
        return

    ss.nr_attempts_total += 1
    ss.nr_current_attempts += 1
    ss.nr_last_choice = choice_index

    # track response time for this question
    if ss.nr_q_started_at is not None:
        response_time = time.time() - ss.nr_q_started_at
    else:
        response_time = 0.0

    ss.nr_resp_times.append(response_time)
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
        "kind": "answer",
        "is_correct": correct,
        "msg": "Correct!" if correct else "Incorrect",
        "correct_label": correct_label,
        "explain": ss.nr_q["explain"],
    }
    # Transition immediately, then perform logging.
    start_feedback_pause()
    suppress_autorefresh()

    selected_answer = ""
    try:
        selected_answer = str(ss.nr_q["choices"][choice_index]["label"])
    except Exception:
        selected_answer = ""

    try:
        log_question_attempt(
            correct=correct,
            missed=False,
            response_time=response_time,
            selected_answer=selected_answer,
        )
    except Exception as e:
        st.warning(f"Could not save question attempt: {e}")

def tick_feedback(now: float):
    if ss.nr_state != "feedback":
        return

    if (now - ss.nr_feedback_started_at) >= ss.nr_feedback_seconds:
        end_feedback_pause_and_advance()

def difficulty_score_weight(level: str) -> float:
    level = str(level or "").strip().lower()
    if level == "challenging":
        return 1.00
    if level == "intermediate":
        return 0.50
    return 0.00  # Starter

def compute_round_score(*, accuracy: float, avg_response_time: float, difficulty_level: str) -> int:
    """
    First-pass round score:
    - accuracy is the main driver
    - harder rounds get a bonus
    - faster average response gets a modest bonus
    """
    accuracy = max(0.0, min(1.0, float(accuracy or 0.0)))
    avg_response_time = float(avg_response_time or 0.0)

    # Main component
    accuracy_points = 70.0 * accuracy

    # Difficulty bonus
    diff_weight = difficulty_score_weight(difficulty_level)
    difficulty_bonus = 20.0 * diff_weight

    # Speed bonus: reward faster-than-target average, capped
    target = float(ss.get("nr_question_seconds", QUESTION_SECONDS_DEFAULT))
    if avg_response_time <= 0:
        speed_bonus = 0.0
    else:
        speed_bonus = 10.0 * ((target / avg_response_time) - 1.0)
        speed_bonus = max(0.0, min(10.0, speed_bonus))

    return int(round(accuracy_points + difficulty_bonus + speed_bonus))

def prepare_round_summary():
    """
    Compute and store round summary values before save begins,
    so they are available immediately during the saving_round UI.
    """
    username = ss.get("username", "unknown")
    round_id = f"nr_round_{ss.nr_round_id}"
    round_key = str(ss.get("nr_round_key", "")).strip() or f"{username}|{ss.nr_session_id}|round_{ss.nr_round_id}"

    total_questions = ss.nr_correct_in_round + ss.nr_incorrect_in_round + ss.nr_missed_in_round
    correct = int(ss.nr_correct_in_round)
    incorrect = int(ss.nr_incorrect_in_round)
    missed = int(ss.nr_missed_in_round)
    attempts_total = int(ss.nr_attempts_total)
    completed = bool(correct >= int(ss.nr_segments))
    accuracy = (float(correct) / float(total_questions)) if total_questions > 0 else 0.0
    end_reason = str(ss.get("nr_round_end_reason", "") or "").strip()
    notes = "" if end_reason == "completed" else end_reason
    now = time.time()
    round_time = float(round_time_elapsed_active(now))

    if ss.nr_resp_times:
        average_response_time = float(sum(ss.nr_resp_times) / len(ss.nr_resp_times))
    else:
        average_response_time = 0.0

    difficulty_level = str(ss.get("nr_round_start_difficulty", ss.get("nr_difficulty", "Starter")))
    score = compute_round_score(
        accuracy=accuracy,
        avg_response_time=average_response_time,
        difficulty_level=difficulty_level,
    )

    ss.nr_last_round_score = score

    ss.nr_round_summary = {
        "username": username,
        "round_id": round_id,
        "round_key": round_key,
        "game_name": "NumeRace",
        "questions_served": int(total_questions),
        "correct": correct,
        "incorrect": incorrect,
        "missed": missed,
        "attempts_total": attempts_total,
        "round_time": round_time,
        "average_response_time": average_response_time,
        "accuracy": accuracy,
        "score": score,
        "completed": completed,
        "start_difficulty_mix": difficulty_level,
        "notes": notes,
    }

def log_round_once(total_q: int):
    """
    Log the current round to Google Sheets exactly once per round_id.
    Safe across Streamlit reruns.
    """
    if ss.get("nr_round_end_reason") == "cancelled" and int(total_q) < 4:
        ss.nr_attempt_buffer = []
        ss.nr_last_attempt_flush_error = ""
        return

    round_already_logged = (ss.get("nr_logged_round_id") == ss.nr_round_id)

    if not round_already_logged:
        summary = ss.get("nr_round_summary")

        if not summary:
            prepare_round_summary()
            summary = ss.get("nr_round_summary", {})

        append_numerace_round(
            username=summary.get("username", ss.get("username", "unknown")),
            round_key=summary.get("round_key", ""),
            round_id=summary.get("round_id", f"nr_round_{ss.nr_round_id}"),
            game_name=summary.get("game_name", "NumeRace"),
            questions_served=int(summary.get("questions_served", 0)),
            correct=int(summary.get("correct", 0)),
            incorrect=int(summary.get("incorrect", 0)),
            missed=int(summary.get("missed", 0)),
            attempts_total=int(summary.get("attempts_total", 0)),
            round_time=float(summary.get("round_time", 0.0)),
            average_response_time=float(summary.get("average_response_time", 0.0)),
            accuracy=float(summary.get("accuracy", 0.0)),
            score=int(summary.get("score", 0)),
            completed=bool(summary.get("completed", False)),
            start_difficulty_mix=summary.get("start_difficulty_mix", ""),
            notes=summary.get("notes", ""),
        )

        ss.nr_logged_round_id = ss.nr_round_id

    if ss.get("nr_attempt_buffer"):
        flush_buffered_attempts()

def flush_buffered_attempts():
    buffered = ss.get("nr_attempt_buffer", [])
    if not buffered:
        ss.nr_last_attempt_flush_error = ""
        return 0

    try:
        append_numerace_attempt_rows(buffered)
        ss.nr_attempt_buffer = []
        ss.nr_last_attempt_flush_error = ""
        return 0
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        ss.nr_last_attempt_flush_error = f"{len(buffered)} attempt row(s) pending. Last error: {detail}"
        return len(buffered)

def log_question_attempt(*, correct: bool, missed: bool, response_time: float, selected_answer: str = ""):
    """
    Log one served question to the numerace_attempts sheet.
    Safe to call once per completed question (answered or timed out).
    """
    q = ss.get("nr_q") or {}
    if not q:
        return

    choices = q.get("choices", [])
    correct_idx = q.get("correct_index", None)

    correct_answer = ""
    if isinstance(correct_idx, int) and 0 <= correct_idx < len(choices):
        correct_answer = str(choices[correct_idx].get("label", ""))

    classification = q.get("classification", {}) or {}
    reporting = q.get("reporting", {}) or {}
    tags = classification.get("tags", []) or []

    try:
        import json
        generated_values_json = json.dumps(q.get("generated_values", {}), ensure_ascii=False)
    except Exception:
        generated_values_json = "{}"

    if "nr_attempt_buffer" not in ss or not isinstance(ss.nr_attempt_buffer, list):
        ss.nr_attempt_buffer = []

    question_seq = int(ss.get("nr_question_seq", 0))
    round_key = str(ss.get("nr_round_key", "")).strip() or f"fallback_round_{ss.nr_round_id}"
    attempt_id = f"{round_key}|q{question_seq}"

    row = {
        "attempt_id": attempt_id,
        "username": ss.get("username", "unknown"),
        "round_key": round_key,
        "round_id": f"nr_round_{ss.nr_round_id}",
        "question_seq": question_seq,
        "question_id": q.get("qid", ""),
        "question_title": q.get("title", ""),
        "domain": classification.get("domain", ""),
        "skill": classification.get("skill", ""),
        "subskill": classification.get("subskill", ""),
        "difficulty": q.get("difficulty", ""),
        "mastery_group": reporting.get("mastery_group", ""),
        "correct": bool(correct),
        "missed": bool(missed),
        "attempts_on_question": int(ss.get("nr_current_attempts", 0)),
        "response_time": float(response_time),
        "selected_answer": str(selected_answer or ""),
        "correct_answer": correct_answer,
        "choice_count": int(len(choices)),
        "prompt_text": q.get("prompt", ""),
        "generated_values_json": generated_values_json,
        "tags_csv": ",".join(str(t) for t in tags),
    }

    ss.nr_attempt_buffer.append(row)

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
def numerace_app():

    st.set_page_config(page_title="NumeRace", layout="wide")
    ss_init()

    now = time.time()

    # Only tick/refresh while actively playing/saving
    if ss.nr_state in ("answering", "feedback", "saving_round"):

        # Always schedule refresh (keeps timers/feedback moving)
        if ss.nr_state == "answering":
            st_autorefresh(interval=AUTOREFRESH_ANSWER_MS, key="nr_tick_answer")
        elif ss.nr_state == "feedback":
            st_autorefresh(interval=AUTOREFRESH_FEEDBACK_MS, key="nr_tick_feedback")
        else:
            st_autorefresh(interval=750, key="nr_tick_saving_round")

        # During the short suppression window, we do NOT run logic that can
        # interfere with clicks / immediate transitions.
        if time.time() >= ss.nr_no_refresh_until:
            if ss.nr_state == "answering":
                handle_timeout_if_needed(now)
            elif ss.nr_state == "feedback":
                tick_feedback(now)

    # ------------------------------------------------------------
    # Top header + styles
    # ------------------------------------------------------------
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 0.35rem;
        }

        .nr-topbar{
          padding: 6px 4px 2px 4px;
          background: transparent;
          border: none;
          border-radius: 0;
          box-shadow: none;
        }

        .nr-title {
          font-size: 34px;
          font-weight: 800;
          letter-spacing: 0.5px;
          text-align: center;
          margin-top: -2px;
        }

        .nr-sub {
          font-size: 14px;
          opacity: 0.82;
          margin-top: -6px;
          text-align: center;
        }

        .nr-card{
            border: none;
            border-radius: 0;
            padding: 2px 0 0 0;
            background: transparent;
            box-shadow: none;
        }

        .nr-qwrap {
            max-width: 760px;
            margin-left:auto;
            margin-right:auto;
        }

        .nr-prompt{
            text-align:center;
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .nr-muted { opacity: 0.75; }

        .nr-controls{
          display:flex;
          flex-direction:column;
          gap:0.15rem;
          padding-top: 0;
        }

        .nr-controls-label{
          font-size: 13px;
          font-weight: 700;
          opacity: 0.78;
          margin-bottom: -0.05rem;
        }

        .nr-sep{
          height: 2px;
          width: 100%;
          margin: 12px 0 16px 0;
          background: linear-gradient(
            to right,
            rgba(15,23,42,0.00),
            rgba(15,23,42,0.22),
            rgba(15,23,42,0.28),
            rgba(15,23,42,0.22),
            rgba(15,23,42,0.00)
          );
        }

        .nr-ring{
          width:92px;
          margin: -2px auto 0.05rem auto;
        }

        .nr-ring-svg{
          width:76px;
          height:76px;
        }

        .nr-ring-bg{
          fill:none;
          stroke: rgba(15, 23, 42, 0.12);
          stroke-width: 6;
        }

        .nr-ring-fg{
          fill:none;
          stroke-width: 6;
          stroke-linecap: round;
        }
        
        .nr-ring-text{
          font-size: 18px;
          font-weight: 800;
          fill: rgba(15, 23, 42, 0.92);
        }

        .nr-track-wrap{
          margin: 4px 0 2px 0;
        }

        .nr-track-flags{
          display:flex;
          justify-content: space-between;
          align-items:center;
          padding: 0 2px 6px 2px;
        }

        .nr-flag{
          font-size: 20px;
          line-height: 1;
          opacity: 0.85;
        }

        .nr-track-stage{
          position: relative;
          overflow: visible;
        }

        .nr-track-line{
          height: 10px;
          border-radius: 999px;
          background: rgba(15, 23, 42, 0.12);
          position: relative;
          overflow: hidden;
        }

        .nr-track-line::after{
          content: "";
          position: absolute;
          inset: 0;
          border-radius: 999px;
          pointer-events: none;
          background-image: repeating-linear-gradient(
            to right,
            transparent 0,
            transparent calc(12.5% - 1px),
            rgba(255,255,255,0.70) calc(12.5% - 1px),
            rgba(255,255,255,0.70) calc(12.5% + 1px)
          );
        }

        .nr-track-fill{
          height: 100%;
          border-radius: 999px;
          background: linear-gradient(90deg,#3b82f6,#2563eb);
        }

        .nr-racer{
          position:absolute;
          top:-16px;
          transform: translateX(-50%) scaleX(-1);
          font-size: 36px;
          transition: left 0.45s ease;
          will-change: left, transform;
          z-index: 5;
          animation: nr-racer-idle 1.35s ease-in-out infinite;
        }

        .nr-racer-finish{
          animation:
            nr-racer-idle 1.35s ease-in-out infinite,
            nr-finish-burst 0.75s ease-out 1;
        }

        .nr-smoke{
          position:absolute;
          top:-6px;
          transform: translateX(-50%);
          font-size: 22px;
          opacity: 0.82;
          z-index: 4;
          pointer-events: none;
          animation: nr-smoke-puff 0.9s ease-in-out infinite;
        }

        @keyframes nr-racer-idle{
          0%   { transform: translateX(-50%) scaleX(-1) translateY(0px); }
          50%  { transform: translateX(-50%) scaleX(-1) translateY(-2px); }
          100% { transform: translateX(-50%) scaleX(-1) translateY(0px); }
        }

        @keyframes nr-smoke-puff{
          0%   { transform: translateX(-50%) scale(0.92); opacity: 0.45; }
          50%  { transform: translateX(-50%) translateX(-2px) scale(1.06); opacity: 0.82; }
          100% { transform: translateX(-50%) translateX(-4px) scale(1.18); opacity: 0.20; }
        }

        @keyframes nr-finish-burst{
          0%   { transform: translateX(-50%) scaleX(-1) translateY(0px) scale(1); }
          35%  { transform: translateX(-50%) scaleX(-1) translateY(-5px) scale(1.22) rotate(5deg); }
          65%  { transform: translateX(-50%) scaleX(-1) translateY(-2px) scale(1.12) rotate(-4deg); }
          100% { transform: translateX(-50%) scaleX(-1) translateY(0px) scale(1.0); }
        }

        .nr-ring-danger .nr-ring-svg{
          animation: nr-timer-pulse 0.85s ease-in-out infinite;
          transform-origin: center;
        }

        @keyframes nr-timer-pulse{
          0%   { transform: rotate(-90deg) scale(1); }
          50%  { transform: rotate(-90deg) scale(1.06); }
          100% { transform: rotate(-90deg) scale(1); }
        }

        .nr-score-pop{
          text-align:center;
          margin-top:0.35rem;
          margin-bottom:1.0rem;
          padding:0.85rem 0.8rem 0.95rem 0.8rem;
          border-radius:18px;
          background:rgba(255,255,255,0.24);
          border:1px solid rgba(15,23,42,0.10);
          animation: nr-score-pop-in 0.42s ease-out 1;
        }

        .nr-score-kicker{
          font-size:1.0rem;
          opacity:0.72;
          margin-bottom:0.15rem;
        }

        .nr-score-value{
          font-size:3.15rem;
          font-weight:800;
          line-height:1.0;
          letter-spacing:0.5px;
        }

        @keyframes nr-score-pop-in{
          0%   { transform: translateY(8px) scale(0.94); opacity: 0; }
          65%  { transform: translateY(-2px) scale(1.03); opacity: 1; }
          100% { transform: translateY(0px) scale(1.0); opacity: 1; }
        }
        
        .nr-qcount{
          margin: 4px auto 0 auto;
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
        
        .nr-qcount-danger{
          border-color: #ef4444;
          color:#ef4444;
          animation: nr-qcount-pulse 0.8s ease-in-out infinite;
        }

        @keyframes nr-qcount-pulse{
          0%   { transform: scale(1); }
          50%  { transform: scale(1.06); }
          100% { transform: scale(1); }
        }

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

        div[data-testid="stButton"] > button {
          border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ------------------------------------------------------------
    # Top bar: left = title + track, right = controls + timer
    # ------------------------------------------------------------
    st.markdown("<div class='nr-topbar'>", unsafe_allow_html=True)

    top_left, top_gap, top_right = st.columns([0.70, 0.05, 0.25], vertical_alignment="top")

    with top_left:
        st.markdown(
            f"""
            <div class='nr-title'>NumeRace</div>
            <div class='nr-sub'>
                Round #{ss.nr_round_id} | {ss.nr_difficulty} | Questions: {ss.nr_correct_in_round}/{ss.nr_segments}
            </div>
            """,
            unsafe_allow_html=True
        )

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

    with top_gap:
        pass

    with top_right:
        st.markdown("<div class='nr-controls'>", unsafe_allow_html=True)

        # Timer first whenever the round is active or saving
        if ss.nr_state in ("answering", "feedback", "saving_round"):
            st.markdown(
                "<div class='nr-controls-label' style='text-align:center;'>Round timer</div>",
                unsafe_allow_html=True
            )
            render_circular_timer(
                "Round",
                max(0.0, round_time_left(now)),
                ss.nr_round_seconds,
                key="nr_ring_round"
            )

        if ss.nr_state == "idle":
            if st.button("Start round", width="stretch"):
                start_round()
                st.rerun()

            st.markdown("<div class='nr-controls-label'>Difficulty level</div>", unsafe_allow_html=True)
            ss.nr_difficulty = st.selectbox(
                "Difficulty level",
                DIFFICULTY_LEVELS,
                index=DIFFICULTY_LEVELS.index(ss.get("nr_difficulty", "Starter"))
                if ss.get("nr_difficulty", "Starter") in DIFFICULTY_LEVELS else 0,
                key="nr_difficulty_picker",
                label_visibility="collapsed",
            )

        elif ss.nr_state == "round_complete":
            if st.button("Next round", width="stretch"):
                ss.nr_round_id += 1
                reset_round()
                st.rerun()

        elif ss.nr_state == "saving_round":
            st.markdown(
                "<div class='nr-controls-label' style='text-align:center;'>Saving round...</div>",
                unsafe_allow_html=True
            )

        else:
            if st.button("Cancel round", width="stretch"):
                ss.nr_round_end_reason = "cancelled"
                ss.nr_feedback = None
                ss.nr_feedback_started_at = None
                ss.nr_round_pause_started_at = None

                total_q = ss.nr_correct_in_round + ss.nr_incorrect_in_round + ss.nr_missed_in_round

                if total_q < 4:
                    # Too short to record anything
                    ss.nr_attempt_buffer = []
                    ss.nr_last_attempt_flush_error = ""
                    prepare_round_summary()
                    ss.nr_state = "round_complete"
                else:
                    prepare_round_summary()
                    ss.nr_state = "saving_round"

                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='nr-sep'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Center question container
    # ------------------------------------------------------------
    st.markdown("<div class='nr-card'><div class='nr-qwrap'>", unsafe_allow_html=True)

    # Idle screen (no start button here - it's in the header)
    if ss.nr_state == "idle":
        if ss.get("nr_start_error"):
            st.error(ss.nr_start_error)
        st.markdown("</div></div>", unsafe_allow_html=True)
        return

    elif ss.nr_state == "saving_round":
        total_q_save = ss.nr_correct_in_round + ss.nr_incorrect_in_round + ss.nr_missed_in_round
        score = ss.get("nr_last_round_score", None)
        if score is not None:
            st.markdown(
                f"""
                <div class="nr-score-pop">
                    <div class="nr-score-kicker">Score</div>
                    <div class="nr-score-value">{score}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown("<div class='nr-prompt'>Saving round...</div>", unsafe_allow_html=True)

        st.markdown(
            "<div class='nr-muted' style='text-align:center;'>Recording results...</div>",
            unsafe_allow_html=True
        )

        try:
            log_round_once(total_q_save)

            pending = len(ss.get("nr_attempt_buffer", []))
            if pending == 0:
                ss.nr_state = "round_complete"
                st.rerun()
            else:
                if ss.get("nr_last_attempt_flush_error"):
                    st.warning(ss.nr_last_attempt_flush_error)

        except Exception as e:
            st.warning(f"Could not save to Google Sheets: {e}")

        st.markdown("</div></div>", unsafe_allow_html=True)
        return

    # Round complete summary (Total / Correct / Incorrect / Missed)
    elif ss.nr_state == "round_complete":

        st.markdown("<div class='nr-prompt'>Round complete!</div>", unsafe_allow_html=True)

        total_q = ss.nr_correct_in_round + ss.nr_incorrect_in_round + ss.nr_missed_in_round
        score = ss.get("nr_last_round_score", None)
        difficulty_label = str(ss.get("nr_round_start_difficulty", ss.get("nr_difficulty", "Starter")))

        if total_q > 0:
            accuracy_pct = round(100.0 * ss.nr_correct_in_round / total_q, 1)
        else:
            accuracy_pct = 0.0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total questions", total_q)
        with c2:
            st.metric("Correct", ss.nr_correct_in_round)
        with c3:
            st.metric("Incorrect", ss.nr_incorrect_in_round)
        with c4:
            st.metric("Missed", ss.nr_missed_in_round)

        c5, c6, c7 = st.columns(3)
        with c5:
            st.metric("Accuracy", f"{accuracy_pct}%")
        with c6:
            st.metric("Difficulty", difficulty_label)
        with c7:
            st.metric("Score", score if score is not None else "—")

        st.markdown(
            "<div class='nr-muted' style='text-align:center;'>Press <b>Next round</b> in the header when ready.</div>",
            unsafe_allow_html=True
        )

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
            st.button(
                ch["label"],
                key=f"nr_choice_{ss.nr_round_id}_{q['qid']}_{i}",
                width="stretch",
                disabled=disabled,
                on_click=submit_answer,
                args=(i,),
            )

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
                head = "[OK] Correct"
            elif ss.nr_last_choice == i:
                cls = "nr-fb nr-fb-bad"
                head = "[X] Your choice"
            else:
                cls = "nr-fb nr-fb-neu"
                head = "[ ]"

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
            f"<div style='text-align:center; font-size:16px; margin-top:12px;'><b>{fb.get('msg', '')}</b> | continuing in {remaining:0.1f}s...</div>",
            unsafe_allow_html=True,
        )


