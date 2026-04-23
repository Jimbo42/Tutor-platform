from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from streamlit import session_state as ss


# ============================================================
# Fraction Visual Practice - DataFrame Model
# ============================================================
# Important note:
# This version uses pandas DataFrames / Styler for the visual fraction tables.
# The matching-total tables use single-cell selection so a clicked cell
# determines how many cells should be shaded.
# ============================================================


# -----------------------------
# Problem model
# -----------------------------
@dataclass
class FractionProblem:
    a_num: int
    a_den: int
    b_num: int
    b_den: int
    operation: str = "add"

    @property
    def a(self) -> Fraction:
        return Fraction(self.a_num, self.a_den)

    @property
    def b(self) -> Fraction:
        return Fraction(self.b_num, self.b_den)

    @property
    def lcd(self) -> int:
        return math.lcm(self.a_den, self.b_den)

    @property
    def a_rows_needed(self) -> int:
        return self.lcd // self.a_den

    @property
    def b_rows_needed(self) -> int:
        return self.lcd // self.b_den

    @property
    def a_equiv_num(self) -> int:
        return self.a_num * self.a_rows_needed

    @property
    def b_equiv_num(self) -> int:
        return self.b_num * self.b_rows_needed

    @property
    def total(self) -> Fraction:
        if getattr(self, "operation", "add") == "subtract":
            return self.a - self.b
        return self.a + self.b

    @property
    def total_num_lcd(self) -> int:
        if getattr(self, "operation", "add") == "subtract":
            return self.a_equiv_num - self.b_equiv_num
        return self.a_equiv_num + self.b_equiv_num

    @property
    def simplest(self) -> Fraction:
        return Fraction(self.total_num_lcd, self.lcd)


def _build_fraction_choices(
    *,
    denominators: tuple[int, ...],
    max_total: Fraction,
) -> list[tuple[str, int, int, int, int]]:
    choices: list[tuple[str, int, int, int, int]] = []
    seen: set[tuple[str, tuple[int, int], tuple[int, int]]] = set()

    for a_den in denominators:
        for b_den in denominators:
            if a_den == b_den:
                continue

            lcd = math.lcm(a_den, b_den)
            if lcd > 24:
                continue

            for a_num in range(1, a_den):
                if math.gcd(a_num, a_den) != 1:
                    continue

                for b_num in range(1, b_den):
                    if math.gcd(b_num, b_den) != 1:
                        continue

                    a_frac = Fraction(a_num, a_den)
                    b_frac = Fraction(b_num, b_den)

                    add_total = a_frac + b_frac
                    if add_total <= max_total:
                        add_key = ("add",) + tuple(sorted(((a_num, a_den), (b_num, b_den))))
                        if add_key not in seen:
                            seen.add(add_key)
                            choices.append(("add", a_num, a_den, b_num, b_den))

                    subtract_total = a_frac - b_frac
                    if Fraction(0, 1) <= subtract_total <= max_total:
                        subtract_key = ("subtract", (a_num, a_den), (b_num, b_den))
                        if subtract_key not in seen:
                            seen.add(subtract_key)
                            choices.append(("subtract", a_num, a_den, b_num, b_den))

    return choices


LEVEL_1_FRACTION_CHOICES = _build_fraction_choices(
    denominators=(2, 3, 4, 5, 6, 8, 9, 10, 12),
    max_total=Fraction(3, 2),
)

LEVEL_2_FRACTION_CHOICES = _build_fraction_choices(
    denominators=(2, 3, 4, 5, 6, 8, 9, 10, 12),
    max_total=Fraction(23, 12),
)


# -----------------------------
# Problem generation
# -----------------------------
def generate_fraction_problem(level: int = 1) -> FractionProblem:
    if level == 1:
        choices = LEVEL_1_FRACTION_CHOICES
    else:
        choices = LEVEL_2_FRACTION_CHOICES

    operation, a_num, a_den, b_num, b_den = random.choice(choices)
    return FractionProblem(a_num, a_den, b_num, b_den, operation)


# -----------------------------
# Session state
# -----------------------------
def _ensure_fraction_state() -> None:
    if "frac_problem" not in ss:
        reset_fraction_activity()
        return
    if not hasattr(ss.frac_problem, "operation"):
        ss.frac_problem.operation = "add"

    if "frac_match_message" not in ss:
        ss.frac_match_message = ""
    if "frac_widget_version" not in ss:
        ss.frac_widget_version = 0
    if "frac_symbolic_status" not in ss:
        ss.frac_symbolic_status = None
    if "frac_lowest_status" not in ss:
        ss.frac_lowest_status = None
    if "frac_mixed_status" not in ss:
        ss.frac_mixed_status = None
    if "frac_symbolic_checked_value" not in ss:
        ss.frac_symbolic_checked_value = ""
    if "frac_lowest_checked_value" not in ss:
        ss.frac_lowest_checked_value = ""
    if "frac_mixed_checked_value" not in ss:
        ss.frac_mixed_checked_value = ""


def reset_fraction_activity(problem: Optional[FractionProblem] = None) -> None:
    if problem is None:
        problem = generate_fraction_problem(level=1)

    ss.frac_widget_version = ss.get("frac_widget_version", 0) + 1
    ss.frac_problem = problem
    ss.frac_feedback = ""
    ss.frac_match_message = ""

    ss.frac_left_rows = 1
    ss.frac_right_rows = 1

    ss.frac_left_selected_end = None
    ss.frac_right_selected_end = None
    ss.frac_sum_selected_end = None

    ss.frac_symbolic = ""
    ss.frac_lowest = ""
    ss.frac_mixed = ""
    ss.frac_symbolic_status = None
    ss.frac_lowest_status = None
    ss.frac_mixed_status = None
    ss.frac_symbolic_checked_value = ""
    ss.frac_lowest_checked_value = ""
    ss.frac_mixed_checked_value = ""


# -----------------------------
# Utilities
# -----------------------------
def _set_feedback(msg: str) -> None:
    ss.frac_feedback = msg


def _set_match_message(msg: str) -> None:
    ss.frac_match_message = msg


def _visible_answer_status(status_key: str, checked_value_key: str, current_value: str) -> Optional[bool]:
    status = ss.get(status_key)
    checked_value = ss.get(checked_value_key, "")
    if status is None or str(current_value or "") != str(checked_value or ""):
        return None
    return bool(status)


def render_answer_status_icon(status: Optional[bool]) -> None:
    if status is True:
        symbol = "✓"
        color = "#2b8a3e"
        bg = "#d3f9d8"
        border = "#8ce99a"
    elif status is False:
        symbol = "✕"
        color = "#c92a2a"
        bg = "#ffe3e3"
        border = "#ffa8a8"
    else:
        symbol = "&nbsp;"
        color = "transparent"
        bg = "transparent"
        border = "transparent"

    st.markdown(
        (
            "<div style='display:flex; justify-content:center; align-items:center; "
            "height:2.5rem; margin-top:0.1rem;'>"
            f"<div style='width:2rem; height:2rem; display:flex; justify-content:center; align-items:center; "
            f"border-radius:999px; border:1px solid {border}; background:{bg}; color:{color}; "
            "font-size:1.1rem; font-weight:700;'>"
            f"{symbol}"
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )


def mixed_number_parts(frac: Fraction) -> Tuple[int, int, int]:
    whole = frac.numerator // frac.denominator
    rem = frac.numerator % frac.denominator
    return whole, rem, frac.denominator


def parse_fraction_text(text: str) -> Optional[Fraction]:
    raw = str(text).strip().replace(" ", "")
    if not raw:
        return None

    try:
        if "/" in raw:
            n_str, d_str = raw.split("/", 1)
            n = int(n_str)
            d = int(d_str)
            if d == 0:
                return None
            return Fraction(n, d)
        return Fraction(int(raw), 1)
    except Exception:
        return None


def parse_mixed_text(text: str) -> Optional[Fraction]:
    raw = str(text).strip().replace("  ", " ")
    if not raw:
        return None

    try:
        if " " in raw:
            whole_str, frac_str = raw.split(" ", 1)
            whole = int(whole_str)
            frac = parse_fraction_text(frac_str)
            if frac is None:
                return None
            return Fraction(whole * frac.denominator + frac.numerator, frac.denominator)
        return parse_fraction_text(raw)
    except Exception:
        return None


# -----------------------------
# DataFrame rendering helpers
# -----------------------------
def make_numbered_df(rows: int, cols: int, start_number: int = 1) -> pd.DataFrame:
    values = []
    n = start_number
    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append(n)
            n += 1
        values.append(row)

    # Use unique internal column names for Styler compatibility.
    col_names = [f"c{j}" for j in range(cols)]
    return pd.DataFrame(values, columns=col_names)

def render_fraction_df(
    *,
    rows: int,
    cols: int,
    filled_count: int,
    color: str,
    title: str = "",
    key: str,
    start_number: int = 1,
) -> None:
    if title:
        st.markdown(
            f"<div style='margin:0 0 0.2rem 0; font-weight:600;'>{title}</div>",
            unsafe_allow_html=True,
        )

    df = make_numbered_df(rows, cols, start_number=start_number)
    cells_html = []
    flat_index = 0

    for row in df.values.tolist():
        row_cells = []
        for value in row:
            bg = color if flat_index < filled_count else "#f8f9fa"
            row_cells.append(
                "<td "
                "style='"
                f"background-color:{bg};"
                "color:#0f172a;"
                "border:1px solid #adb5bd;"
                "text-align:center;"
                "font-weight:600;"
                "padding:0;"
                "height:42px;"
                f"width:{100 / max(cols, 1)}%;"
                "'>"
                f"{value}"
                "</td>"
            )
            flat_index += 1
        cells_html.append("<tr>" + "".join(row_cells) + "</tr>")

    table_html = (
        f"<div id='{key}' style='width:100%; margin:0;'>"
        "<table style='width:100%; border-collapse:collapse; table-layout:fixed; margin:0;'>"
        "<tbody>"
        + "".join(cells_html)
        + "</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_selectable_fraction_df(
    *,
    rows: int,
    cols: int,
    filled_count: int,
    color: str,
    title: str,
    key: str,
    chosen_end: Optional[int],
    selected_state_key: str,
    target_end: Optional[int],
    label_prefix: str,
    start_number: int = 1,
    chunk_size: Optional[int] = None,
    fill_mode: str = "correct_only",
    feedback_target: str = "match",
    display_total: Optional[int] = None,
) -> Optional[int]:
    total = rows * cols
    message_total = display_total or total
    options = list(range(start_number, start_number + total))
    selected_end = ss.get(selected_state_key, chosen_end)

    if title:
        st.markdown(
            (
                "<div style='font-weight:700; margin:0 0 0.02rem 0; line-height:1.1;'>"
                f"{title}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    if chunk_size is None:
        chunk_size = total

    if fill_mode == "selected":
        fill_end = chosen_end or 0
    else:
        fill_end = target_end if target_end is not None and chosen_end == target_end else 0

    button_styles = []
    st.markdown(
        f"""
        <style>
        .st-key-{key}_rows div[data-testid="stVerticalBlock"] {{
            gap: 0;
        }}
        .st-key-{key}_rows div[data-testid="column"] {{
            padding-top: 0;
            padding-bottom: 0;
        }}
        .st-key-{key}_rows div[data-testid="stButton"] {{
            margin-top: 0;
            margin-bottom: 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    for value in options:
        button_key = f"{key}_btn_{value}"
        offset = value - start_number
        row_idx = offset // max(chunk_size, 1)
        col_idx = offset % max(chunk_size, 1)
        if fill_end and value <= fill_end:
            bg = color
            border = color
        elif chosen_end == value:
            bg = "#e9ecef"
            border = "#868e96"
        else:
            bg = "#ffffff"
            border = "#adb5bd"
        button_styles.append(
            f"""
            .st-key-{button_key} button {{
                width: 100%;
                min-height: 44px;
                border-radius: 0;
                border: 1px solid {border};
                background: {bg};
                color: #0f172a;
                font-weight: 600;
                box-shadow: none;
                padding: 0;
                margin: {'-1px 0 0 0' if row_idx > 0 else '0'};
                position: relative;
                left: {'-1px' if col_idx > 0 else '0'};
            }}
            .st-key-{button_key} button:hover {{
                border: 1px solid {border};
                color: #0f172a;
            }}
            """
        )

    st.markdown(
        "<style>" + "\n".join(button_styles) + "</style>",
        unsafe_allow_html=True,
    )

    with st.container(key=f"{key}_rows"):
        for start in range(0, len(options), chunk_size):
            chunk_options = options[start : start + chunk_size]
            row_key = f"{key}_row_{start}"
            row_margin = "-0.5rem" if start > 0 else "0"
            st.markdown(
                f"""
                <style>
                .st-key-{row_key} {{
                    margin-top: {row_margin};
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

            with st.container(key=row_key):
                button_cols = st.columns(len(chunk_options), gap=None)

                for idx, value in enumerate(chunk_options):
                    button_key = f"{key}_btn_{value}"
                    with button_cols[idx]:
                        if st.button(str(value), key=button_key, width="stretch"):
                            ss[selected_state_key] = value
                            if target_end is not None and value == target_end:
                                if feedback_target == "match":
                                    _set_match_message("")
                                else:
                                    _set_feedback(f"Good. You selected the correct {label_prefix}: {value}/{message_total}.")
                            elif target_end is not None:
                                if feedback_target == "match":
                                    _set_match_message(
                                        f"Not quite. You selected {value}/{message_total} for the {label_prefix}."
                                    )
                                else:
                                    _set_feedback(
                                        f"Not quite. You selected {value}/{message_total} for the {label_prefix}."
                                    )
                            st.rerun()

    return ss.get(selected_state_key, chosen_end)


def render_status_badge(state: str, label: str, large: bool = False) -> None:
    colors = {
        "red": ("#ffe3e3", "#c92a2a", "#7f1d1d"),
        "yellow": ("#fff3bf", "#e67700", "#8f5b00"),
        "green": ("#d3f9d8", "#2b8a3e", "#1b5e20"),
    }
    bg, border, text = colors[state]
    size = "1rem" if large else "0.95rem"
    pad = "1rem 1.25rem" if large else "0.7rem 1rem"
    min_width = "180px" if large else "140px"

    st.markdown(
        (
            f"<div style='display:flex; justify-content:center;'>"
            f"<div style='background:{bg}; color:{text}; border:2px solid {border}; border-radius:16px;"
            f"padding:{pad}; min-width:{min_width}; text-align:center; font-size:{size}; font-weight:700;"
            f"line-height:1.35;'>{label}</div>"
            f"</div>"
        ),
        unsafe_allow_html=True,
    )


def render_selector_buttons(
    *,
    key_prefix: str,
    total: int,
    target_end: Optional[int],
    chosen_end: Optional[int],
    enabled: bool,
    label_prefix: str,
) -> None:
    if total <= 0:
        return

    st.caption("Select the ending number for the shaded amount.")

    chunk_size = 8
    for start in range(1, total + 1, chunk_size):
        end = min(start + chunk_size - 1, total)
        cols = st.columns(end - start + 1)
        for i, value in enumerate(range(start, end + 1)):
            btn_label = f"{value}"
            if cols[i].button(btn_label, key=f"{key_prefix}_{value}", width="stretch", disabled=not enabled):
                if target_end is not None and value == target_end:
                    if key_prefix == "left_pick":
                        ss.frac_left_selected_end = value
                    elif key_prefix == "right_pick":
                        ss.frac_right_selected_end = value
                    elif key_prefix == "sum_pick":
                        ss.frac_sum_selected_end = value
                    _set_feedback(f"Good. You selected the correct {label_prefix}: {value}/{total}.")
                else:
                    _set_feedback(f"Not quite. You selected {value}/{total} for the {label_prefix}.")
                st.rerun()

    current = chosen_end if chosen_end is not None else 0
    st.caption(f"Current selection: {current}/{total}")


def render_multiplier_input(side: str) -> None:
    row_key = f"frac_{side}_rows"
    input_key = f"{side}_multiplier_input_{ss.get('frac_widget_version', 0)}"
    current_value = ss[row_key]

    st.markdown(
        (
            "<div style='margin:0 0 0.15rem 0; font-size:0.95rem; "
            "text-align:center;'>Multiply denominator by:</div>"
        ),
        unsafe_allow_html=True,
    )

    left_pad, input_col, right_pad = st.columns([1, 2, 1])
    with input_col:
        new_value = st.number_input(
            "Multiply denominator by:",
            min_value=1,
            step=1,
            value=current_value,
            key=input_key,
            label_visibility="collapsed",
        )

    if new_value != current_value:
        ss[row_key] = int(new_value)
        ss.frac_match_message = ""
        if side == "left":
            ss.frac_left_selected_end = None
        else:
            ss.frac_right_selected_end = None
        st.rerun()

# -----------------------------
# Section state logic
# -----------------------------
def section2_status(problem: FractionProblem) -> Tuple[str, str]:
    left_total = ss.frac_left_rows * problem.a_den
    right_total = ss.frac_right_rows * problem.b_den

    if left_total != right_total:
        return "red", "Denominators do not match"

    left_ok = left_total == problem.lcd and ss.frac_left_selected_end == problem.a_equiv_num
    right_ok = right_total == problem.lcd and ss.frac_right_selected_end == problem.b_equiv_num

    if left_ok and right_ok:
        return "green", "Equivalent fractions complete"

    return "yellow", "Denominators match — set numerators"


# -----------------------------
# Section renderers
# -----------------------------
def render_section_1(problem: FractionProblem) -> None:
    operator_symbol = "-" if problem.operation == "subtract" else "+"
    c1, c2, c3 = st.columns([5.75, 0.9, 5.75])
    with c1:
        st.markdown(
            (
                "<div style='text-align:center; font-size:1.35rem; font-weight:700; margin:0 0 0.5rem 0;'>"
                f"{problem.a_num}/{problem.a_den}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        render_fraction_df(
            rows=1,
            cols=problem.a_den,
            filled_count=problem.a_num,
            color="#74c0fc",
            key=f"original_left_{problem.a_num}_{problem.a_den}",
            start_number=1,
        )
    with c2:
        st.markdown(
            (
                "<div style='text-align:center; font-size:1.45rem; font-weight:700; "
                f"padding-top:2.45rem;'>{operator_symbol}</div>"
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            (
                "<div style='text-align:center; font-size:1.35rem; font-weight:700; margin:0 0 0.5rem 0;'>"
                f"{problem.b_num}/{problem.b_den}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        render_fraction_df(
            rows=1,
            cols=problem.b_den,
            filled_count=problem.b_num,
            color="#ffa94d",
            key=f"original_right_{problem.b_num}_{problem.b_den}",
            start_number=1,
        )


def render_section_2(problem: FractionProblem) -> None:
    operator_symbol = "-" if problem.operation == "subtract" else "+"
    state, label = section2_status(problem)

    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"] > div {
            gap: 0.2rem;
        }
        div[data-testid="stNumberInput"] {
            margin-bottom: 0 !important;
        }
        div[data-testid="stNumberInput"] input {
            text-align: center;
        }
        div[data-testid="stAlert"] {
            margin-top: 0.25rem;
            margin-bottom: 0.25rem;
            padding-top: 0.4rem;
            padding-bottom: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Row 1: multiplier inputs + status badge
    top_left, top_mid, top_right = st.columns([5.75, 0.9, 5.75], gap="small")

    with top_left:
        render_multiplier_input("left")

    with top_mid:
        st.markdown("<div style='height:0.7rem;'></div>", unsafe_allow_html=True)
        render_status_badge(state, label, large=True)
        if ss.frac_match_message:
            st.warning(ss.frac_match_message)

    with top_right:
        render_multiplier_input("right")

    # Pull the segment row upward so it sits much closer to the controls row.
    st.markdown("<div style='margin-top:-0.6rem;'></div>", unsafe_allow_html=True)

    # Row 2: connected segment tables
    bot_left, bot_mid, bot_right = st.columns([5.75, 0.9, 5.75], gap="small")

    with bot_left:
        left_total = ss.frac_left_rows * problem.a_den
        ss.frac_left_selected_end = render_selectable_fraction_df(
            rows=ss.frac_left_rows,
            cols=problem.a_den,
            filled_count=ss.frac_left_selected_end or 0,
            color="#74c0fc",
            title="",
            key=f"left_match_{ss.frac_left_rows}_{problem.a_den}",
            chosen_end=ss.frac_left_selected_end,
            selected_state_key="frac_left_selected_end",
            target_end=problem.a_equiv_num if left_total == problem.lcd else None,
            label_prefix="first equivalent fraction",
        )

    with bot_mid:
        st.markdown("<div style='height:0.1rem;'></div>", unsafe_allow_html=True)

    with bot_right:
        right_total = ss.frac_right_rows * problem.b_den
        ss.frac_right_selected_end = render_selectable_fraction_df(
            rows=ss.frac_right_rows,
            cols=problem.b_den,
            filled_count=ss.frac_right_selected_end or 0,
            color="#ffa94d",
            title="",
            key=f"right_match_{ss.frac_right_rows}_{problem.b_den}",
            chosen_end=ss.frac_right_selected_end,
            selected_state_key="frac_right_selected_end",
            target_end=problem.b_equiv_num if right_total == problem.lcd else None,
            label_prefix="second equivalent fraction",
        )

    label_left, label_mid, label_right = st.columns([5.75, 0.9, 5.75], gap="small")

    with label_left:
        st.markdown(
            (
                "<div style='text-align:center; font-weight:700; margin-top:0.25rem;'>"
                f"Equivalent Fraction: {ss.frac_left_selected_end or 0}/{left_total}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    with label_mid:
        if state == "green":
            st.markdown(
                (
                    "<div style='text-align:center; font-size:1.7rem; font-weight:700; "
                    f"margin-top:0.05rem;'>{operator_symbol}</div>"
                ),
                unsafe_allow_html=True,
            )

    with label_right:
        st.markdown(
            (
                "<div style='text-align:center; font-weight:700; margin-top:0.25rem;'>"
                f"Equivalent Fraction: {ss.frac_right_selected_end or 0}/{right_total}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_section_3(problem: FractionProblem) -> None:
    state, _ = section2_status(problem)
    if state != "green":
        return

    show_mixed_input = problem.total > 1
    if not show_mixed_input:
        ss.frac_mixed = ""

    sum_rows = 2 if problem.total > 1 else 1

    ss.frac_sum_selected_end = render_selectable_fraction_df(
        rows=sum_rows,
        cols=problem.lcd,
        filled_count=ss.frac_sum_selected_end or 0,
        color="#c0eb75",
        title="Select the numerator total",
        key=f"sum_pick_{problem.lcd}",
        chosen_end=ss.frac_sum_selected_end,
        selected_state_key="frac_sum_selected_end",
        target_end=problem.total_num_lcd,
        label_prefix="sum",
        chunk_size=problem.lcd,
        fill_mode="selected",
        feedback_target="main",
        display_total=problem.lcd,
    )

    sum_is_correct = ss.frac_sum_selected_end == problem.total_num_lcd
    if ss.frac_sum_selected_end is not None and not sum_is_correct:
        st.warning(f"Invalid selection. You selected {ss.frac_sum_selected_end}/{problem.lcd}.")

    if not sum_is_correct:
        return

    st.markdown("#### Symbolic answers")
    if show_mixed_input:
        columns = st.columns([4, 0.7, 4, 0.7, 4, 0.7, 2.2], gap="small")
        symbolic_col, symbolic_icon_col, lowest_col, lowest_icon_col, mixed_col, mixed_icon_col, button_col = columns
    else:
        columns = st.columns([4, 0.7, 4, 0.7, 2.2], gap="small")
        symbolic_col, symbolic_icon_col, lowest_col, lowest_icon_col, button_col = columns
        mixed_col = None
        mixed_icon_col = None

    with symbolic_col:
        ss.frac_symbolic = st.text_input(
            "Fraction shown",
            key="frac_symbolic_widget",
            autocomplete="off",
        )
    with symbolic_icon_col:
        render_answer_status_icon(
            _visible_answer_status("frac_symbolic_status", "frac_symbolic_checked_value", ss.frac_symbolic)
        )

    with lowest_col:
        ss.frac_lowest = st.text_input(
            "Lowest equivalent fraction",
            key="frac_lowest_widget",
            autocomplete="off",
        )
    with lowest_icon_col:
        render_answer_status_icon(
            _visible_answer_status("frac_lowest_status", "frac_lowest_checked_value", ss.frac_lowest)
        )

    if show_mixed_input:
        with mixed_col:
            whole, rem, den = mixed_number_parts(problem.total)
            mixed_placeholder = f"{whole} {rem}/{den}" if rem else str(whole)
            ss.frac_mixed = st.text_input(
                "Mixed fraction",
                key="frac_mixed_widget",
                autocomplete="off",
            )
        with mixed_icon_col:
            render_answer_status_icon(
                _visible_answer_status("frac_mixed_status", "frac_mixed_checked_value", ss.frac_mixed)
            )

    with button_col:
        st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)
        if st.button("Check answers", key="check_fraction_answers", width="stretch"):
            symbolic = parse_fraction_text(ss.frac_symbolic)
            ss.frac_symbolic_status = symbolic == Fraction(problem.total_num_lcd, problem.lcd)
            ss.frac_symbolic_checked_value = ss.frac_symbolic

            lowest = parse_fraction_text(ss.frac_lowest)
            ss.frac_lowest_status = lowest == problem.simplest
            ss.frac_lowest_checked_value = ss.frac_lowest

            if show_mixed_input:
                mixed = parse_mixed_text(ss.frac_mixed)
                ss.frac_mixed_status = mixed == problem.total
                ss.frac_mixed_checked_value = ss.frac_mixed
            else:
                ss.frac_mixed_status = None
                ss.frac_mixed_checked_value = ""

            _set_feedback("")
            st.rerun()

    if False and st.button("Check answers", key="check_fraction_answers", width="stretch"):
        notes = []

        notes.append("sum model ✓" if ss.frac_sum_selected_end == problem.total_num_lcd else "sum model ✗")

        symbolic = parse_fraction_text(ss.frac_symbolic)
        notes.append(
            "symbolic fraction ✓"
            if symbolic == Fraction(problem.total_num_lcd, problem.lcd)
            else "symbolic fraction ✗"
        )

        lowest = parse_fraction_text(ss.frac_lowest)
        notes.append(
            "lowest equivalent fraction ✓"
            if lowest == problem.simplest
            else "lowest equivalent fraction ✗"
        )

        mixed = parse_mixed_text(ss.frac_mixed)
        notes.append("mixed fraction ✓" if mixed == problem.total else "mixed fraction ✗")

        if not show_mixed_input:
            notes = [note for note in notes if not note.startswith("mixed fraction")]

        _set_feedback(" | ".join(notes))
        st.rerun()


# -----------------------------
# Main renderer
# -----------------------------
def render_fraction_visual_practice() -> None:
    _ensure_fraction_state()
    problem: FractionProblem = ss.frac_problem

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-left: 0.8rem;
            padding-right: 0.8rem;
            max-width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    title_col, new_problem_col, reset_problem_col = st.columns([6, 1.5, 1.8], gap="small")
    with title_col:
        st.markdown("### Fraction Practice")
    with new_problem_col:
        st.markdown("<div style='margin-top: 0.35rem;'></div>", unsafe_allow_html=True)
        if st.button("New problem", key="new_fraction_problem", width="stretch"):
            reset_fraction_activity()
            st.rerun()
    with reset_problem_col:
        st.markdown("<div style='margin-top: 0.35rem;'></div>", unsafe_allow_html=True)
        if st.button("Reset this problem", key="reset_fraction_problem", width="stretch"):
            reset_fraction_activity(problem)
            st.rerun()

    render_section_1(problem)
    st.markdown(
        "<hr style='margin:0.35rem 0 0.2rem 0; border:none; border-top:1px solid rgba(15, 23, 42, 0.12);'>",
        unsafe_allow_html=True,
    )
    render_section_2(problem)
    st.markdown(
        "<hr style='margin:0.3rem 0 0.15rem 0; border:none; border-top:1px solid rgba(15, 23, 42, 0.12);'>",
        unsafe_allow_html=True,
    )
    render_section_3(problem)
