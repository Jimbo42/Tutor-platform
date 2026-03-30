# numerace.py
import time
import streamlit as st
from streamlit import session_state as ss
from streamlit_autorefresh import st_autorefresh
from pathlib import Path
import sys
import uuid
import random

# Ensure project root is on sys.path so we can import from /shared
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.numeracy_dsl import load_game, pick_question_def, build_question
from shared.google_db import (
    append_numerace_round,
    append_numerace_attempt_rows,
    get_numerace_user_profile_rows_with_row_numbers,
    upsert_numerace_user_profile_rows_fast,
)
# ------------------------------------------------------------
# Config (tune these later / per difficulty)
# ------------------------------------------------------------
SEGMENTS_PER_ROUND = 8

QUESTION_SECONDS_DEFAULT = 10        # time per question
FEEDBACK_SECONDS_DEFAULT = 3.5      # paused feedback display time (excluded from round timer)
DIFFICULTY_LEVELS = ["Starter", "Intermediate", "Challenging"]

AUTOREFRESH_FEEDBACK_MS = 250  # fast: snappy auto-advance
AUTOREFRESH_SAVING_MS = 2000

RECENT_WINDOW = 8
MIN_EVIDENCE_FOR_ADAPT = 5

ACTION_MULTIPLIERS = {
    "stretch": 0.70,
    "maintain": 1.00,
    "fluency_practice": 1.20,
    "stabilize": 1.35,
    "support": 1.60,
}

ACC_HIGH = 0.85
ACC_LOW = 0.60
FAST_FACTOR = 0.75
SLOW_FACTOR = 1.25
DEFAULT_RECOMMENDED_ACTION = "maintain"
DEFAULT_MULTIPLIER = 1.00

# ------------------------------------------------------------
# Helpers: session-state init
# ------------------------------------------------------------
def ss_init():
    if "nr_initialized" in ss:
        return

    ss.nr_initialized = True

    ss.nr_segments = SEGMENTS_PER_ROUND
    ss.nr_question_seconds = QUESTION_SECONDS_DEFAULT
    ss.nr_feedback_seconds = FEEDBACK_SECONDS_DEFAULT

    ss.nr_round_id = 1
    ss.nr_correct_in_round = 0

    # idle | answering | feedback | saving_round | round_complete
    ss.nr_state = "idle"

    # question timing only
    ss.nr_q_started_at = None
    ss.nr_q_elapsed_frozen = 0.0

    # current question payload
    ss.nr_q = None
    ss.nr_last_choice = None
    ss.nr_feedback = None
    ss.nr_feedback_started_at = None
    ss.nr_no_refresh_until = 0.0

    # stats
    ss.nr_attempts_total = 0
    ss.nr_correct_total = 0
    ss.nr_difficulty = "Starter"
    ss.nr_start_error = None
    ss.nr_round_start_difficulty = ss.nr_difficulty
    ss.nr_round_end_reason = ""

    # round breakdown
    ss.nr_incorrect_in_round = 0

    # totals breakdown
    ss.nr_incorrect_total = 0

    # DSL game config
    ss.nr_game_path = str(PROJECT_ROOT / "shared" / "numeracy_game.json")
    ss.nr_game = load_game(ss.nr_game_path)

    # selection tracking within a round
    ss.nr_q_history = []
    ss.nr_q_used_counts = {}

    # pull defaults from JSON rules
    rules = ss.nr_game.get("rules", {})
    ss.nr_segments = int(rules.get("segments_per_round", ss.nr_segments))
    ss.nr_question_seconds = int(rules.get("question_seconds", ss.nr_question_seconds))
    ss.nr_feedback_seconds = int(rules.get("feedback_seconds", ss.nr_feedback_seconds))

    ss.nr_logged_round_id = None
    ss.nr_resp_times = []
    ss.nr_question_seq = 0
    ss.nr_current_attempts = 0
    ss.nr_attempt_buffer = []
    ss.nr_last_attempt_flush_error = ""

    # stable ids for idempotent logging
    ss.nr_session_id = uuid.uuid4().hex[:12]
    ss.nr_round_key = ""
    ss.nr_last_round_score = None
    ss.nr_score_toast_shown_for_round = None
    ss.nr_round_summary = None
    ss.nr_attempt_flush_started_for_round = None
    ss.nr_attempt_flush_done_for_round = None
    ss.nr_profile_updated_for_round = None

    ss.nr_profile_map = {}
    ss.nr_profile_loaded_for_user = None
    ss.nr_recent_cache = {}
    ss.nr_recent_cache_loaded_for_user = None
    ss.nr_profile_row_map = {}

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return int(default)

def _profile_key(row: dict):
    return (
        str(row.get("username", "")).strip(),
        str(row.get("domain", "")).strip(),
        str(row.get("skill", "")).strip(),
        str(row.get("subskill", "")).strip(),
        str(row.get("mastery_group", "")).strip(),
    )

def load_user_profile_into_session(force: bool = False):
    username = str(ss.get("username", "unknown")).strip() or "unknown"

    if (not force) and ss.get("nr_profile_loaded_for_user") == username and isinstance(ss.get("nr_profile_map"), dict):
        return ss.nr_profile_map

    rows = get_numerace_user_profile_rows_with_row_numbers(username=username) or []
    profile_map = {}
    profile_row_map = {}

    for row in rows:
        key = _profile_key(row)
        profile_map[key] = row
        profile_row_map[key] = row.get("_row_number")

    ss.nr_profile_map = profile_map
    ss.nr_profile_row_map = profile_row_map
    ss.nr_profile_loaded_for_user = username
    ss.nr_recent_cache_loaded_for_user = None
    return profile_map

def _recent_cache_key_from_parts(username: str, domain: str, skill: str, subskill: str, mastery_group: str):
    return (
        str(username or "").strip(),
        str(domain or "").strip(),
        str(skill or "").strip(),
        str(subskill or "").strip(),
        str(mastery_group or "").strip(),
    )

def _recent_cache_key_from_attempt_row(row: dict):
    return _recent_cache_key_from_parts(
        row.get("username", ss.get("username", "unknown")),
        row.get("domain", ""),
        row.get("skill", ""),
        row.get("subskill", ""),
        row.get("mastery_group", ""),
    )

def build_recent_cache_from_profile_map():
    """
    Build a lightweight rolling cache from the current profile rows.
    Since the sheet only stores recent aggregates, not the actual last 8 attempts,
    we seed the cache with synthetic entries that preserve the same recent stats.
    """
    username = str(ss.get("username", "unknown")).strip() or "unknown"
    cache = {}

    profile_map = ss.get("nr_profile_map") or {}
    for key, row in profile_map.items():
        if not isinstance(row, dict):
            continue

        recent_accuracy = _safe_float(row.get("recent_accuracy", 0.0), 0.0)
        recent_avg_time = _safe_float(row.get("recent_avg_response_time", 0.0), 0.0)

        n_seed = RECENT_WINDOW
        n_correct = int(round(recent_accuracy * n_seed))
        n_correct = max(0, min(n_seed, n_correct))
        n_incorrect = n_seed - n_correct

        entries = (
            [{"correct": True, "response_time": recent_avg_time}] * n_correct
            + [{"correct": False, "response_time": recent_avg_time}] * n_incorrect
        )

        cache[key] = entries[-RECENT_WINDOW:]

    ss.nr_recent_cache = cache
    ss.nr_recent_cache_loaded_for_user = username
    return cache

def ensure_recent_cache_loaded(force: bool = False):
    username = str(ss.get("username", "unknown")).strip() or "unknown"

    if (
        not force
        and ss.get("nr_recent_cache_loaded_for_user") == username
        and isinstance(ss.get("nr_recent_cache"), dict)
    ):
        return ss.nr_recent_cache

    return build_recent_cache_from_profile_map()

def get_profile_row_for_question(qdef: dict):
    username = str(ss.get("username", "unknown")).strip() or "unknown"
    classification = (qdef or {}).get("classification", {}) or {}
    reporting = (qdef or {}).get("reporting", {}) or {}

    key = (
        username,
        str(classification.get("domain", "")).strip(),
        str(classification.get("skill", "")).strip(),
        str(classification.get("subskill", "")).strip(),
        str(reporting.get("mastery_group", "")).strip(),
    )
    return (ss.get("nr_profile_map") or {}).get(key)

def classify_speed_band(avg_time: float, target_time: float) -> str:
    if avg_time <= 0 or target_time <= 0:
        return "unknown"
    if avg_time <= FAST_FACTOR * target_time:
        return "fast"
    if avg_time > SLOW_FACTOR * target_time:
        return "slow"
    return "ok"

def recommend_action(recent_accuracy: float, recent_avg_time: float, target_time: float) -> str:
    speed_band = classify_speed_band(recent_avg_time, target_time)

    if recent_accuracy >= ACC_HIGH:
        if speed_band == "fast":
            return "stretch"
        if speed_band == "slow":
            return "fluency_practice"
        return "maintain"

    if recent_accuracy < ACC_LOW:
        if speed_band == "slow":
            return "support"
        return "stabilize"

    return "maintain"

def build_speed_accuracy_band(recent_accuracy: float, recent_avg_time: float, target_time: float) -> str:
    speed_band = classify_speed_band(recent_avg_time, target_time)
    acc_band = "correctish" if recent_accuracy >= ACC_HIGH else "incorrectish" if recent_accuracy < ACC_LOW else "mixed"
    return f"{speed_band}_{acc_band}"

def summarize_attempt_group(rows: list[dict], target_time: float) -> dict:
    rows = rows or []
    n = len(rows)

    correct_count = sum(1 for r in rows if bool(r.get("correct", False)))
    incorrect_count = n - correct_count
    total_time = sum(_safe_float(r.get("response_time", 0.0)) for r in rows)
    total_attempts = sum(_safe_int(r.get("attempts_on_question", 0)) for r in rows)

    accuracy = (correct_count / n) if n else 0.0
    avg_time = (total_time / n) if n else 0.0
    avg_attempts = (total_attempts / n) if n else 0.0

    if n < MIN_EVIDENCE_FOR_ADAPT:
        action = DEFAULT_RECOMMENDED_ACTION
        multiplier = DEFAULT_MULTIPLIER
    else:
        action = recommend_action(accuracy, avg_time, target_time)
        multiplier = ACTION_MULTIPLIERS.get(action, DEFAULT_MULTIPLIER)

    return {
        "questions_seen": n,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "accuracy": accuracy,
        "avg_response_time": avg_time,
        "avg_attempts_on_question": avg_attempts,
        "recommended_action": action,
        "current_multiplier": multiplier,
        "speed_accuracy_band": build_speed_accuracy_band(accuracy, avg_time, target_time),
    }

def update_user_profile_from_round():
    """
    Incrementally update only the profile rows touched in the current round.
    Uses in-memory profile rows plus an in-memory rolling recent cache.
    No attempt-history re-reads.
    """
    username = str(ss.get("username", "unknown")).strip() or "unknown"
    attempt_rows = list(ss.get("nr_attempt_buffer", []) or [])

    if not attempt_rows:
        return

    load_user_profile_into_session(force=False)
    ensure_recent_cache_loaded(force=False)

    profile_map = ss.get("nr_profile_map") or {}
    recent_cache = ss.get("nr_recent_cache") or {}
    target_time = float(ss.get("nr_question_seconds", QUESTION_SECONDS_DEFAULT))

    grouped = {}
    for r in attempt_rows:
        key = _recent_cache_key_from_attempt_row(r)
        grouped.setdefault(key, []).append(r)

    profile_updates = []

    for key, rows in grouped.items():
        _, domain, skill, subskill, mastery_group = key
        prev = dict(profile_map.get(key, {}) or {})

        prev_questions_seen = _safe_int(prev.get("questions_seen", 0), 0)
        prev_correct_count = _safe_int(prev.get("correct_count", 0), 0)
        prev_incorrect_count = _safe_int(prev.get("incorrect_count", 0), 0)

        batch_n = len(rows)
        batch_correct = sum(1 for r in rows if bool(r.get("correct", False)))
        batch_incorrect = batch_n - batch_correct
        batch_total_time = sum(_safe_float(r.get("response_time", 0.0), 0.0) for r in rows)

        new_questions_seen = prev_questions_seen + batch_n
        new_correct_count = prev_correct_count + batch_correct
        new_incorrect_count = prev_incorrect_count + batch_incorrect
        new_accuracy = (new_correct_count / new_questions_seen) if new_questions_seen else 0.0

        prev_avg_time = _safe_float(prev.get("avg_response_time", 0.0), 0.0)
        prev_total_time = prev_avg_time * prev_questions_seen
        new_total_time = prev_total_time + batch_total_time
        new_avg_time = (new_total_time / new_questions_seen) if new_questions_seen else 0.0

        existing_recent = list(recent_cache.get(key, []) or [])
        for r in rows:
            existing_recent.append({
                "correct": bool(r.get("correct", False)),
                "response_time": _safe_float(r.get("response_time", 0.0), 0.0),
            })
        existing_recent = existing_recent[-RECENT_WINDOW:]
        recent_cache[key] = existing_recent

        recent_n = len(existing_recent)
        recent_correct = sum(1 for x in existing_recent if bool(x.get("correct", False)))
        recent_accuracy = (recent_correct / recent_n) if recent_n else 0.0
        recent_avg_time = (
            sum(_safe_float(x.get("response_time", 0.0), 0.0) for x in existing_recent) / recent_n
            if recent_n else 0.0
        )

        if recent_n < MIN_EVIDENCE_FOR_ADAPT:
            recommended_action = DEFAULT_RECOMMENDED_ACTION
            current_multiplier = DEFAULT_MULTIPLIER
        else:
            recommended_action = recommend_action(recent_accuracy, recent_avg_time, target_time)
            current_multiplier = ACTION_MULTIPLIERS.get(recommended_action, DEFAULT_MULTIPLIER)

        last_seen = ""
        if rows:
            last_seen = str(rows[-1].get("timestamp", ""))

        row = {
            "timestamp_updated": last_seen,
            "username": username,
            "domain": domain,
            "skill": skill,
            "subskill": subskill,
            "mastery_group": mastery_group,
            "questions_seen": new_questions_seen,
            "correct_count": new_correct_count,
            "missed_count": 0,
            "incorrect_count": new_incorrect_count,
            "accuracy": new_accuracy,
            "avg_response_time": new_avg_time,
            "recent_accuracy": recent_accuracy,
            "recent_avg_response_time": recent_avg_time,
            "current_multiplier": current_multiplier,
            "recommended_action": recommended_action,
            "last_seen": last_seen,
        }

        profile_map[key] = row
        profile_updates.append(row)

    ss.nr_profile_map = profile_map
    ss.nr_recent_cache = recent_cache

    if profile_updates:
        updated_row_map = upsert_numerace_user_profile_rows_fast(
            profile_updates,
            existing_row_map=ss.get("nr_profile_row_map", {}) or {},
        )
        ss.nr_profile_row_map = updated_row_map

def render_solid_track(correct: int, segments: int):
    segs = max(1, int(segments))
    correct_i = max(0, int(correct))

    prog = clamp01(correct_i / segs)
    pct = int(round(prog * 100))

    racer_pct = max(4, min(96, pct))

    finished = correct_i >= segs
    extra = "nr-racer-finish" if finished else ""
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

def suppress_autorefresh(seconds: float = 0.5):
    ss.nr_no_refresh_until = time.time() + seconds

def start_round():
    now = time.time()
    ss.nr_start_error = None
    ss.nr_round_start_difficulty = ss.get("nr_difficulty", "Starter")
    ss.nr_round_end_reason = ""

    username = str(ss.get("username", "unknown")).strip() or "unknown"
    ss.nr_round_key = f"{username}|{ss.nr_session_id}|round_{ss.nr_round_id}"

    ss.nr_state = "answering"
    ss.nr_q_elapsed_frozen = 0.0
    ss.nr_q_started_at = now

    ss.nr_incorrect_in_round = 0

    ss.nr_q_history = []
    ss.nr_q_used_counts = {}

    ss.nr_logged_round_id = None
    ss.nr_resp_times = []
    ss.nr_question_seq = 0
    ss.nr_current_attempts = 0
    ss.nr_attempt_buffer = []
    ss.nr_last_attempt_flush_error = ""
    ss.nr_attempt_flush_started_for_round = None
    ss.nr_attempt_flush_done_for_round = None
    ss.nr_profile_updated_for_round = None

    load_user_profile_into_session(force=False)
    ensure_recent_cache_loaded(force=False)

    try:
        ss.nr_q = make_question()
    except Exception as e:
        ss.nr_state = "idle"
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
    ss.nr_state = "idle"

    ss.nr_q_elapsed_frozen = 0.0
    ss.nr_q_started_at = None
    ss.nr_q = None

    ss.nr_last_choice = None
    ss.nr_feedback = None
    ss.nr_feedback_started_at = None

    ss.nr_incorrect_in_round = 0

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
    ss.nr_attempt_flush_started_for_round = None
    ss.nr_attempt_flush_done_for_round = None
    ss.nr_profile_updated_for_round = None

def start_feedback_pause():
    ss.nr_state = "feedback"
    ss.nr_feedback_started_at = time.time()

def end_feedback_pause_and_advance():
    now = time.time()

    if ss.nr_correct_in_round >= ss.nr_segments:
        ss.nr_round_end_reason = "completed"
        prepare_round_summary()
        ss.nr_state = "saving_round"
        ss.nr_feedback = None
        ss.nr_feedback_started_at = None
        return

    ss.nr_state = "answering"
    ss.nr_q_started_at = now
    ss.nr_q_elapsed_frozen = 0.0
    ss.nr_q = make_question()

    suppress_autorefresh()

    ss.nr_last_choice = None
    ss.nr_feedback = None
    ss.nr_feedback_started_at = None

# ------------------------------------------------------------
# Question generation
# ------------------------------------------------------------
def pick_question_def_adaptive(game: dict, q_history: list, q_used_counts: dict, difficulty: str):
    questions = list((game or {}).get("questions", []) or [])
    questions = [q for q in questions if q.get("enabled", True)]

    if difficulty:
        questions = [q for q in questions if str(q.get("difficulty", "")).strip() == str(difficulty).strip()]

    if not questions:
        raise ValueError(f"No enabled questions found for difficulty '{difficulty}'")

    cooldown_default = _safe_int((game.get("selection", {}) or {}).get("cooldown_default", 1), 1)
    eligible = []

    for q in questions:
        qid = str(q.get("id", "")).strip()
        sel = q.get("selection", {}) or {}
        reporting = q.get("reporting", {}) or {}

        cooldown = _safe_int(sel.get("cooldown", cooldown_default), cooldown_default)
        max_per_round = _safe_int(sel.get("max_per_round", 999), 999)
        base_weight = max(0.01, _safe_float(sel.get("weight", 1.0), 1.0))

        used_count = _safe_int(q_used_counts.get(qid, 0), 0)
        if used_count >= max_per_round:
            continue

        if cooldown > 0 and qid in q_history[-cooldown:]:
            continue

        profile_row = get_profile_row_for_question(q)
        multiplier = DEFAULT_MULTIPLIER
        if profile_row:
            multiplier = max(0.10, _safe_float(profile_row.get("current_multiplier", 1.0), 1.0))

        effective_weight = base_weight * multiplier
        eligible.append((q, effective_weight))

    if not eligible:
        # fall back to original selector behavior
        return pick_question_def(game, q_history, q_used_counts, difficulty=difficulty)

    total_weight = sum(w for _, w in eligible)
    r = random.random() * total_weight
    upto = 0.0
    for q, w in eligible:
        upto += w
        if upto >= r:
            return q

    return eligible[-1][0]

def make_question():

    qdef = pick_question_def_adaptive(
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
def submit_answer(choice_index: int):
    if ss.nr_state != "answering":
        return

    ss.nr_attempts_total += 1
    ss.nr_current_attempts += 1
    ss.nr_last_choice = choice_index

    if ss.nr_q_started_at is not None:
        response_time = max(0.0, time.time() - ss.nr_q_started_at)
    else:
        response_time = 0.0

    ss.nr_resp_times.append(response_time)
    ss.nr_q_elapsed_frozen = response_time

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
        "elapsed": response_time,
    }

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
            response_time=response_time,
            selected_answer=selected_answer,
        )
    except Exception as e:
        st.warning(f"Could not save question attempt: {e}")

def tick_feedback(now: float):
    if ss.nr_state != "feedback":
        return

    if ss.nr_feedback_started_at is None:
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

    total_questions = ss.nr_correct_in_round + ss.nr_incorrect_in_round
    correct = int(ss.nr_correct_in_round)
    incorrect = int(ss.nr_incorrect_in_round)
    attempts_total = int(ss.nr_attempts_total)
    completed = bool(correct >= int(ss.nr_segments))
    accuracy = (float(correct) / float(total_questions)) if total_questions > 0 else 0.0
    end_reason = str(ss.get("nr_round_end_reason", "") or "").strip()
    notes = "" if end_reason == "completed" else end_reason

    round_time = float(sum(ss.nr_resp_times)) if ss.nr_resp_times else 0.0
    average_response_time = (round_time / len(ss.nr_resp_times)) if ss.nr_resp_times else 0.0

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
        "missed": 0,
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
    round_id = ss.get("nr_round_id")

    if ss.get("nr_round_end_reason") == "cancelled" and int(total_q) < 4:
        ss.nr_attempt_buffer = []
        ss.nr_last_attempt_flush_error = ""
        ss.nr_attempt_flush_started_for_round = None
        ss.nr_attempt_flush_done_for_round = None
        ss.nr_profile_updated_for_round = None
        return

    round_already_logged = (ss.get("nr_logged_round_id") == round_id)

    if not round_already_logged:
        summary = ss.get("nr_round_summary")

        if not summary:
            prepare_round_summary()
            summary = ss.get("nr_round_summary", {})

        append_numerace_round(
            username=summary.get("username", ss.get("username", "unknown")),
            round_key=summary.get("round_key", ""),
            round_id=summary.get("round_id", f"nr_round_{round_id}"),
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

        ss.nr_logged_round_id = round_id

    # Keep a copy of the rows touched in this round before the flush claims the buffer.
    buffered_attempts = list(ss.get("nr_attempt_buffer", []) or [])

    pending = flush_buffered_attempts()

    # Only update profile once, and only after the attempts are safely flushed.
    if (
        pending == 0
        and buffered_attempts
        and ss.get("nr_profile_updated_for_round") != round_id
    ):
        ss.nr_attempt_buffer = buffered_attempts
        try:
            update_user_profile_from_round()
            ss.nr_profile_updated_for_round = round_id
        finally:
            ss.nr_attempt_buffer = []

def flush_buffered_attempts():
    round_id = ss.get("nr_round_id")

    # already flushed successfully for this round
    if ss.get("nr_attempt_flush_done_for_round") == round_id:
        ss.nr_last_attempt_flush_error = ""
        return 0

    buffered = list(ss.get("nr_attempt_buffer", []) or [])
    if not buffered:
        ss.nr_last_attempt_flush_error = ""
        return 0

    # If another rerun already claimed this round's buffer, do not send again.
    if ss.get("nr_attempt_flush_started_for_round") == round_id:
        return len(buffered)

    # Claim the buffer immediately so reruns cannot resend it.
    ss.nr_attempt_flush_started_for_round = round_id
    ss.nr_attempt_buffer = []

    try:
        append_numerace_attempt_rows(buffered)
        ss.nr_attempt_flush_done_for_round = round_id
        ss.nr_last_attempt_flush_error = ""
        return 0
    except Exception as e:
        # Restore the buffer on failure so it can be retried safely.
        ss.nr_attempt_buffer = buffered
        ss.nr_attempt_flush_started_for_round = None
        detail = f"{type(e).__name__}: {e}"
        ss.nr_last_attempt_flush_error = f"{len(buffered)} attempt row(s) pending. Last error: {detail}"
        return len(buffered)

def log_question_attempt(*, correct: bool, response_time: float, selected_answer: str = ""):
    """
    Log one served question to the numerace_attempts sheet.
    Safe to call once per answered question.
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
        "missed": False,
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

    ss_init()

    now = time.time()

    # Only refresh during feedback auto-advance or while saving
    if ss.nr_state in ("feedback", "saving_round"):
        if ss.nr_state == "feedback":
            st_autorefresh(interval=AUTOREFRESH_FEEDBACK_MS, key="nr_tick_feedback")
        else:
            st_autorefresh(interval=AUTOREFRESH_SAVING_MS, key="nr_tick_saving_round")

        if time.time() >= ss.nr_no_refresh_until:
            if ss.nr_state == "feedback":
                tick_feedback(now)

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

        render_solid_track(
            ss.nr_correct_in_round,
            ss.nr_segments
        )

    with top_gap:
        pass

    with top_right:
        st.markdown("<div class='nr-controls'>", unsafe_allow_html=True)

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

                total_q = ss.nr_correct_in_round + ss.nr_incorrect_in_round

                if total_q < 4:
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
        total_q_save = ss.nr_correct_in_round + ss.nr_incorrect_in_round
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

        total_q = ss.nr_correct_in_round + ss.nr_incorrect_in_round
        score = ss.get("nr_last_round_score", None)
        difficulty_label = str(ss.get("nr_round_start_difficulty", ss.get("nr_difficulty", "Starter")))

        if total_q > 0:
            accuracy_pct = round(100.0 * ss.nr_correct_in_round / total_q, 1)
        else:
            accuracy_pct = 0.0

        total_question_time = float(sum(ss.nr_resp_times)) if ss.nr_resp_times else 0.0
        avg_question_time = (total_question_time / len(ss.nr_resp_times)) if ss.nr_resp_times else 0.0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total questions", total_q)
        with c2:
            st.metric("Correct", ss.nr_correct_in_round)
        with c3:
            st.metric("Incorrect", ss.nr_incorrect_in_round)
        with c4:
            st.metric("Accuracy", f"{accuracy_pct}%")

        c5, c6, c7 = st.columns(3)
        with c5:
            st.metric("Avg question time", f"{avg_question_time:.1f}s")
        with c6:
            st.metric("Total question time", f"{total_question_time:.1f}s")
        with c7:
            st.metric("Score", score if score is not None else "—")

        st.markdown(
            f"<div class='nr-muted' style='text-align:center;'>Difficulty: <b>{difficulty_label}</b></div>",
            unsafe_allow_html=True
        )

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

        elapsed = float(fb.get("elapsed", ss.get("nr_q_elapsed_frozen", 0.0)))
        remaining = max(0.0, ss.nr_feedback_seconds - (now - ss.nr_feedback_started_at))

        st.markdown(
            f"<div style='text-align:center; font-size:16px; margin-top:12px;'><b>{fb.get('msg', '')}</b> | Time: {elapsed:0.1f}s | continuing in {remaining:0.1f}s...</div>",
            unsafe_allow_html=True,
        )


