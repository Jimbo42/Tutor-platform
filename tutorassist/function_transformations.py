import json
import math
import random
import re
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# Function Transformations MVP for TutorAssist
# -----------------------------------------------------------------------------
# This is a single-file starter module intended for rapid prototyping.
# Once the flow feels right, the next refactor should split this into:
#   - shared/function_transform_engine.py
#   - shared/function_transform_plot.py
#   - shared/function_transform_families.py
#   - tutorassist/function_transformations.py
#
# Supported families in this MVP:
#   - quadratic      y = a(x-h)^2 + k
#   - absolute       y = a|x-h| + k
#   - cubic          y = a(x-h)^3 + k
#   - sine           y = a sin(x-h) + k
#
# Scope deliberately excludes horizontal scaling for now.
# -----------------------------------------------------------------------------

@dataclass
class ProblemInstance:
    family_id: str
    family_name: str
    parent_label: str
    equation_template: str
    x_axis_display: Dict
    parameter_ranges: Dict
    a: float
    h: float
    k: float
    b: float | None
    angle_unit: str | None
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    display_equation: str
    key_points: List[Tuple[float, float, str]]

TRANSFORMS_PATH = Path(__file__).resolve().parent.parent / "shared" / "transforms.json"


def load_family_defs() -> Dict[str, Dict]:
    with TRANSFORMS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    family_defs = data.get("family_defs")
    if not isinstance(family_defs, dict):
        raise ValueError("shared/transforms.json must contain a 'family_defs' object.")
    return family_defs


FAMILY_DEFS: Dict[str, Dict] = load_family_defs()


def choose_parameter_value(meta: Dict, parameter_name: str):
    parameter_ranges = meta.get("parameter_ranges", {})
    param_def = parameter_ranges.get(parameter_name)
    values = param_def.get("values") if isinstance(param_def, dict) else None
    if values:
        return random.choice(values)
    if not isinstance(param_def, dict):
        raise ValueError(f"Missing parameter_ranges.{parameter_name} range for family.")
    return random.randint(int(param_def["min"]), int(param_def["max"]))


def choose_x_window(meta: Dict) -> Tuple[float, float]:
    parameter_ranges = meta.get("parameter_ranges", {})
    x_def = parameter_ranges.get("x")
    if not isinstance(x_def, dict):
        raise ValueError("Missing parameter_ranges.x range for family.")
    return float(x_def["min"]), float(x_def["max"])


def get_x_axis_display(meta: Dict) -> Dict:
    x_axis_display = meta.get("x_axis_display", {"mode": "integer"})
    if not isinstance(x_axis_display, dict):
        raise ValueError("x_axis_display must be an object when provided.")
    return x_axis_display

def get_b_value(meta: Dict) -> float | None:
    parameter_ranges = meta.get("parameter_ranges", {})
    if "b" not in parameter_ranges:
        return None
    return float(choose_parameter_value(meta, "b"))


def get_angle_unit(meta: Dict) -> str | None:
    x_axis_display = meta.get("x_axis_display", {})
    mode = x_axis_display.get("mode", "integer")
    if mode in ("radians", "degrees"):
        return mode
    return None


def trig_period_from_b(b: float | None) -> float:
    bb = 1.0 if b is None else float(b)
    return (2 * math.pi) / abs(bb)


def format_shift_value(value: float) -> str:
    if math.isclose(value, 0, abs_tol=1e-9):
        return "0"
    if math.isclose(value / math.pi, round(value / math.pi), abs_tol=1e-9) or math.isclose((value / math.pi) * 2, round((value / math.pi) * 2), abs_tol=1e-9):
        return format_pi_multiple(value)
    return format_numeric_value(value)


def format_shift_inner(h: float) -> str:
    if math.isclose(h, 0, abs_tol=1e-9):
        return "x"
    if h > 0:
        return f"x - {format_shift_value(h)}"
    return f"x + {format_shift_value(abs(h))}"


def format_factor(value: float) -> str:
    if math.isclose(value, 1, abs_tol=1e-9):
        return ""
    if math.isclose(value, -1, abs_tol=1e-9):
        return "-"
    return format_numeric_value(value)

def format_numeric_value(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0, abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:g}"


def format_fraction_value(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0, abs_tol=1e-9):
        return str(int(round(value)))
    fraction = Fraction(value).limit_denominator(12)
    if math.isclose(float(fraction), value, rel_tol=0, abs_tol=1e-9):
        return f"{fraction.numerator}/{fraction.denominator}"
    return format_numeric_value(value)


def format_pi_text(value: float) -> str:
    ratio = value / math.pi
    fraction = Fraction(ratio).limit_denominator(12)
    if not math.isclose(float(fraction), ratio, rel_tol=0, abs_tol=1e-9):
        return format_numeric_value(value)
    if fraction.numerator == 0:
        return "0"
    if fraction.denominator == 1:
        if fraction.numerator == 1:
            return "pi"
        if fraction.numerator == -1:
            return "-pi"
        return f"{fraction.numerator}pi"
    sign = "-" if fraction.numerator < 0 else ""
    numerator = abs(fraction.numerator)
    numerator_text = "" if numerator == 1 else str(numerator)
    return f"{sign}{numerator_text}pi/{fraction.denominator}"


def format_pi_multiple(value: float) -> str:
    ratio = value / math.pi
    best_numerator = None
    best_denominator = None
    for denominator in (1, 2, 3, 4, 6, 8):
        numerator = round(ratio * denominator)
        if math.isclose(ratio, numerator / denominator, rel_tol=0, abs_tol=1e-9):
            best_numerator = int(numerator)
            best_denominator = denominator
            break
    if best_numerator is None or best_denominator is None:
        return format_numeric_value(value)
    numerator = best_numerator
    denominator = best_denominator
    if numerator == 0:
        return "0"
    if numerator % denominator == 0:
        integer = numerator // denominator
        if integer == 1:
            return "π"
        if integer == -1:
            return "-π"
        return f"{integer}π"
    sign = "-" if numerator < 0 else ""
    return f"{sign}{abs(numerator)}π/{denominator}"


# -----------------------------------------------------------------------------
# Page setup / styling
# -----------------------------------------------------------------------------

def inject_transform_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }
        .tf-title {
            margin-top: 0;
            margin-bottom: 0.1rem;
            font-size: 2rem;
            font-weight: 700;
        }
        .tf-header-actions {
            padding-top: 0.35rem;
        }
        .tf-subtitle {
            margin-top: 0;
            margin-bottom: 1rem;
            max-width: 44rem;
            font-weight: 600;
            line-height: 1.35;
            color: rgba(49, 51, 63, 0.75);
        }
        .tf-card {
            border: 0.5px solid rgba(49, 51, 63, 0.12);
            border-radius: 0.9rem;
            padding: 0.9rem 0.85rem;
            background: rgba(255,255,255,0.72);
            margin-bottom: 0.8rem;
        }
        .tf-graph {
            padding: 0;
            margin-bottom: 0.35rem;
        }
        .tf-small {
            font-size: 0.95rem;
            color: rgba(49, 51, 63, 0.8);
        }
        .tf-good {
            color: #0f7b0f;
            font-weight: 600;
        }
        .tf-bad {
            color: #aa2222;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Session state helpers
# -----------------------------------------------------------------------------

def init_state() -> None:
    defaults = {
        "tf_problem": None,
        "tf_points": [],
        "tf_point_text": "",
        "tf_point_label": "P",
        "tf_point_error": "",
        "tf_parent_choice": None,
        "tf_accepted_parent": None,
        "tf_a_guess": 1,
        "tf_h_guess": 0,
        "tf_h_guess_text": "0",
        "tf_k_guess": 0,
        "tf_b_guess": None,
        "tf_equation_input": "",
        "tf_available_hints": [],
        "tf_hints_shown": [],
        "tf_hint_view_index": -1,
        "tf_parent_submitted": False,
        "tf_parent_correct": False,
        "tf_params_submitted": False,
        "tf_params_correct": False,
        "tf_equation_submitted": False,
        "tf_equation_correct": False,
        "tf_feedback": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# -----------------------------------------------------------------------------
# Problem generation
# -----------------------------------------------------------------------------

def generate_problem() -> ProblemInstance:
    family_id = random.choice(list(FAMILY_DEFS.keys()))
    meta = FAMILY_DEFS[family_id]
    x_axis_display = get_x_axis_display(meta)

    a = float(choose_parameter_value(meta, "a"))
    h = float(choose_parameter_value(meta, "h"))
    k = float(choose_parameter_value(meta, "k"))
    b = get_b_value(meta)
    angle_unit = get_angle_unit(meta)

    x_min, x_max = choose_x_window(meta)

    xs = np.linspace(x_min, x_max, 800)
    ys = evaluate_family(family_id, xs, a, h, k, b)
    finite_ys = ys[np.isfinite(ys)]

    raw_min = float(np.min(finite_ys))
    raw_max = float(np.max(finite_ys))

    y_min = max(-25, math.floor(raw_min) - 2)
    y_max = min(25, math.ceil(raw_max) + 2)
    if y_max - y_min < 10:
        center = (y_max + y_min) / 2
        y_min = math.floor(center - 5)
        y_max = math.ceil(center + 5)

    display_equation = build_display_equation(family_id, a, h, k, b)
    key_points = build_key_points(family_id, a, h, k, b)

    return ProblemInstance(
        family_id=family_id,
        family_name=meta["family_name"],
        parent_label=meta["parent_label"],
        equation_template=meta["equation_template"],
        x_axis_display=x_axis_display,
        parameter_ranges=meta.get("parameter_ranges", {}),
        a=a,
        h=h,
        k=k,
        b=b,
        angle_unit=angle_unit,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        display_equation=display_equation,
        key_points=key_points,
    )

# -----------------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------------

def evaluate_family(family_id: str, x, a: float, h: float, k: float, b: float | None = None):
    x = np.array(x, dtype=float)

    if family_id == "quadratic":
        return a * (x - h) ** 2 + k

    if family_id == "absolute":
        return a * np.abs(x - h) + k

    if family_id == "cubic":
        return a * (x - h) ** 3 + k

    if family_id == "square_root":
        inner = x - h
        y = np.full_like(inner, np.nan, dtype=float)
        valid = inner >= 0
        y[valid] = a * np.sqrt(inner[valid]) + k
        return y

    if family_id == "reciprocal":
        y = np.full_like(x, np.nan, dtype=float)
        valid = np.abs(x - h) > 1e-9
        y[valid] = a / (x[valid] - h) + k
        return y

    if family_id == "sin":
        bb = 1.0 if b is None else float(b)
        return a * np.sin(bb * (x - h)) + k

    if family_id == "cos":
        bb = 1.0 if b is None else float(b)
        return a * np.cos(bb * (x - h)) + k

    raise ValueError(f"Unsupported family_id: {family_id}")


def build_display_equation(family_id: str, a: float, h: float, k: float, b: float | None = None) -> str:
    a_str = "" if math.isclose(a, 1, abs_tol=1e-9) else "-" if math.isclose(a, -1, abs_tol=1e-9) else format_numeric_value(a)
    inner = format_shift_inner(h)
    k_str = f" + {format_numeric_value(k)}" if k > 0 else f" - {format_numeric_value(abs(k))}" if k < 0 else ""

    if family_id == "quadratic":
        return f"y = {a_str}({inner})²{k_str}" if inner != "x" else f"y = {a_str}x²{k_str}"

    if family_id == "absolute":
        return f"y = {a_str}|{inner}|{k_str}"

    if family_id == "cubic":
        return f"y = {a_str}({inner})³{k_str}" if inner != "x" else f"y = {a_str}x³{k_str}"

    if family_id == "square_root":
        return f"y = {a_str}sqrt({inner}){k_str}" if inner != "x" else f"y = {a_str}sqrt(x){k_str}"

    if family_id == "reciprocal":
        return f"y = {a_str}/({inner}){k_str}" if inner != "x" else f"y = {a_str}/x{k_str}"

    if family_id in ("sin", "cos"):
        func = "sin" if family_id == "sin" else "cos"
        bb = 1.0 if b is None else float(b)
        b_str = format_factor(bb)

        if inner == "x":
            inner_expr = "x"
        else:
            inner_expr = f"({inner})"

        trig_inner = f"{b_str}{inner_expr}" if b_str else inner_expr
        return f"y = {a_str}{func}({trig_inner}){k_str}"

    return ""

def build_key_points(family_id: str, a: float, h: float, k: float, b: float | None = None) -> List[Tuple[float, float, str]]:
    # Label strings are for instructor/debug use later if needed.
    if family_id == "quadratic":
        pts = [
            (h, k, "vertex"),
            (h + 1, a + k, "right 1"),
            (h - 1, a + k, "left 1"),
            (h + 2, 4 * a + k, "right 2"),
            (h - 2, 4 * a + k, "left 2"),
        ]
        return pts

    if family_id == "absolute":
        pts = [
            (h, k, "vertex"),
            (h + 1, abs(a) + k if a > 0 else -abs(a) + k, "right 1"),
            (h - 1, abs(a) + k if a > 0 else -abs(a) + k, "left 1"),
            (h + 2, 2 * abs(a) + k if a > 0 else -2 * abs(a) + k, "right 2"),
            (h - 2, 2 * abs(a) + k if a > 0 else -2 * abs(a) + k, "left 2"),
        ]
        return pts

    if family_id == "cubic":
        pts = [
            (h, k, "center"),
            (h + 1, a + k, "right 1"),
            (h - 1, -a + k, "left 1"),
            (h + 2, 8 * a + k, "right 2"),
            (h - 2, -8 * a + k, "left 2"),
        ]
        return pts

    if family_id == "square_root":
        pts = [
            (h, k, "start"),
            (h + 1, a + k, "right 1"),
            (h + 4, 2 * a + k, "right 4"),
            (h + 9, 3 * a + k, "right 9"),
        ]
        return pts

    if family_id == "reciprocal":
        pts = [
            (h + 1, a + k, "right 1"),
            (h + 2, a / 2 + k, "right 2"),
            (h - 1, -a + k, "left 1"),
            (h - 2, -a / 2 + k, "left 2"),
        ]
        return pts

    if family_id == "sin":
        cycle = trig_period_from_b(b)
        pts = [
            (h, k, "midline rise"),
            (h + cycle / 4, a + k, "max"),
            (h + cycle / 2, k, "midline fall"),
            (h + 3 * cycle / 4, -a + k, "min"),
            (h + cycle, k, "period end"),
        ]
        return pts

    if family_id == "cos":
        cycle = trig_period_from_b(b)
        pts = [
            (h, a + k, "max"),
            (h + cycle / 4, k, "midline fall"),
            (h + cycle / 2, -a + k, "min"),
            (h + 3 * cycle / 4, k, "midline rise"),
            (h + cycle, a + k, "period end"),
        ]
        return pts

    return []


def format_point(x: float, y: float) -> str:
    return f"({x:g}, {y:g})"


def parse_point_text(raw_text: str) -> Tuple[float, float]:
    cleaned = raw_text.strip()
    match = re.fullmatch(r"\(?\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)?", cleaned)
    if not match:
        raise ValueError("Enter the point as (x,y) or x,y.")
    return float(match.group(1)), float(match.group(2))


def add_point_from_inputs() -> None:
    point_text = st.session_state.tf_point_text
    label = st.session_state.tf_point_label
    try:
        x_val, y_val = parse_point_text(point_text)
    except ValueError as exc:
        st.session_state.tf_point_error = str(exc)
        return

    st.session_state.tf_points.append({"x": x_val, "y": y_val, "label": label or "P"})
    st.session_state.tf_point_text = ""
    st.session_state.tf_point_label = "P"
    st.session_state.tf_point_error = ""


def clear_points() -> None:
    st.session_state.tf_points = []
    st.session_state.tf_point_text = ""
    st.session_state.tf_point_label = "P"
    st.session_state.tf_point_error = ""


def normalize_points_rows(rows: List[Dict]) -> List[Dict]:
    updated_points = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            x_val = float(row.get("x", 0))
            y_val = float(row.get("y", 0))
        except (TypeError, ValueError):
            continue
        updated_points.append(
            {
                "x": x_val,
                "y": y_val,
                "label": str(row.get("label", "P") or "P"),
            }
        )
    return updated_points


def format_radian_tick(value: float) -> str:
    return format_pi_multiple(value)


def build_radian_label_alias(problem: ProblemInstance) -> Dict[float, str]:
    aliases: Dict[float, str] = {}
    half_pi = math.pi / 2
    start_index = math.ceil(problem.x_min / half_pi)
    end_index = math.floor(problem.x_max / half_pi)
    for index in range(start_index, end_index + 1):
        value = index * half_pi
        aliases[value] = format_radian_tick(value)
    return aliases

def build_degree_ticks(problem: ProblemInstance) -> tuple[list[float], list[str]]:
    x_span = problem.x_max - problem.x_min
    if x_span <= 180:
        step = 15
    elif x_span <= 360:
        step = 30
    else:
        step = 45

    first_tick = math.ceil(problem.x_min / step) * step
    tickvals = []
    ticktext = []
    value = first_tick
    while value <= problem.x_max + 1e-9:
        tickvals.append(value)
        ticktext.append(f"{int(round(value))}°")
        value += step
    return tickvals, ticktext


def build_radian_ticks(problem: ProblemInstance) -> tuple[list[float], list[str]]:
    display = getattr(problem, "x_axis_display", {}) or {}
    num = int(display.get("step_numerator", 1))
    den = int(display.get("step_denominator", 2))
    step = (num * math.pi) / den

    first_i = math.ceil(problem.x_min / step)
    last_i = math.floor(problem.x_max / step)

    tickvals = []
    ticktext = []
    for i in range(first_i, last_i + 1):
        val = i * step
        tickvals.append(val)
        ticktext.append(format_pi_multiple(val))
    return tickvals, ticktext


def build_x_axis_config(problem: ProblemInstance) -> Dict:
    display = getattr(problem, "x_axis_display", None) or {"mode": "integer"}
    mode = display.get("mode", "integer")

    base = {
        "range": [problem.x_min, problem.x_max],
        "showgrid": True,
        "zeroline": False,
    }

    if mode == "integer":
        return {
            **base,
            "tickmode": "auto",
        }

    if mode == "degrees":
        tickvals, ticktext = build_degree_ticks(problem)
        return {
            **base,
            "tickmode": "array",
            "tickvals": tickvals,
            "ticktext": ticktext,
        }

    if mode == "radians":
        tickvals, ticktext = build_radian_ticks(problem)
        return {
            **base,
            "tickmode": "array",
            "tickvals": tickvals,
            "ticktext": ticktext,
        }

    return {
        **base,
        "tickmode": "auto",
    }

# -----------------------------------------------------------------------------
# Plot builder
# -----------------------------------------------------------------------------

def build_figure(problem: ProblemInstance, student_points: List[Dict]) -> go.Figure:
    xs = np.linspace(problem.x_min, problem.x_max, 700)
    if problem.family_id == "square_root":
        xs = np.sort(np.unique(np.append(xs, problem.h)))
    fig = go.Figure()
    if problem.family_id == "reciprocal":
        gap = max((problem.x_max - problem.x_min) / 700, 1e-3)
        left_x = xs[xs < problem.h - gap]
        right_x = xs[xs > problem.h + gap]
        left_y = evaluate_family(problem.family_id, left_x, problem.a, problem.h, problem.k, getattr(problem, "b", None))
        right_y = evaluate_family(problem.family_id, right_x, problem.a, problem.h, problem.k, getattr(problem, "b", None))
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([left_x, [np.nan], right_x]),
                y=np.concatenate([left_y, [np.nan], right_y]),
                mode="lines",
                name="Function",
                line=dict(width=2, color="#1f77b4"),
                hoverinfo="skip",
                connectgaps=False,
            )
        )
    else:
        ys = evaluate_family(problem.family_id, xs, problem.a, problem.h, problem.k, getattr(problem, "b", None))
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name="Function",
                hoverinfo="skip",
            )
        )

    # Axes lines
    fig.add_hline(y=0, line_width=1)
    fig.add_vline(x=0, line_width=1)

    if student_points:
        px = [p["x"] for p in student_points]
        py = [p["y"] for p in student_points]
        labels = [p["label"] for p in student_points]
        fig.add_trace(
            go.Scatter(
                x=px,
                y=py,
                mode="markers+text",
                text=labels,
                textposition="top center",
                name="Student points",
            )
        )

    if problem.family_id == "reciprocal":
        fig.add_trace(
            go.Scatter(
                x=[problem.h, problem.h],
                y=[problem.y_min, problem.y_max],
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        dragmode="pan",
    )
    fig.update_xaxes(**build_x_axis_config(problem))
    fig.update_yaxes(
        range=[problem.y_min, problem.y_max],
        showgrid=True,
        zeroline=False,
        constrain="range",
        tickmode="auto",
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------

def check_parent(problem: ProblemInstance, selected_parent: str) -> Tuple[bool, str]:
    if problem.family_id in ("sin", "cos") and selected_parent in ("sin", "cos"):
        return True, "Good. A sine or cosine form can represent this graph depending on the phase shift."

    if selected_parent == problem.family_id:
        return True, f"Correct. The matching transformation form is {problem.equation_template}."

    return False, "That transformation form does not match the graph. Focus on the parent shape first."


def format_transformation_choice(family_id: str) -> str:
    forms = {
        "quadratic": r"$y = a(x - h)^2 + k$",
        "absolute": r"$y = a|x - h| + k$",
        "cubic": r"$y = a(x - h)^3 + k$",
        "square_root": r"$y = a\sqrt{x - h} + k$",
        "reciprocal": r"$y = \frac{a}{x - h} + k$",
        "sin": r"$y = a\sin(b(x - h)) + k$",
        "cos": r"$y = a\cos(b(x - h)) + k$",
    }
    return forms[family_id]


def get_effective_parent_problem(problem: ProblemInstance) -> ProblemInstance:
    accepted_parent = st.session_state.get("tf_accepted_parent")
    if accepted_parent not in FAMILY_DEFS or accepted_parent == problem.family_id:
        return problem

    if {accepted_parent, problem.family_id} != {"sin", "cos"}:
        return problem

    meta = FAMILY_DEFS[accepted_parent]
    b_value = 1.0 if problem.b is None else float(problem.b)
    phase_shift = math.pi / (2 * b_value)
    adjusted_h = problem.h - phase_shift if accepted_parent == "sin" else problem.h + phase_shift

    return replace(
        problem,
        family_id=accepted_parent,
        family_name=meta["family_name"],
        parent_label=meta["parent_label"],
        equation_template=meta["equation_template"],
        parameter_ranges=meta.get("parameter_ranges", {}),
        h=adjusted_h,
        display_equation=build_display_equation(accepted_parent, problem.a, adjusted_h, problem.k, problem.b),
    )


def build_hints(problem: ProblemInstance) -> List[str]:
    hints = {
        "quadratic": [
            "Look for the vertex first.",
            "Check whether the parabola opens up or down.",
            "Compare the width to the parent parabola.",
        ],
        "absolute": [
            "Find the corner point first.",
            "Check whether the arms open up or down.",
            "Compare the steepness to the parent absolute-value graph.",
        ],
        "cubic": [
            "Find the center point of the S-shape first.",
            "Check whether the graph rises or falls from left to right.",
            "Compare points one unit from the center to estimate the stretch.",
        ],
        "square_root": [
            "Look for the starting endpoint first.",
            "The square-root graph only extends in one horizontal direction from that point.",
            "Use nearby perfect-square steps to estimate the vertical stretch.",
        ],
        "reciprocal": [
            "Look for the vertical and horizontal asymptotes first.",
            "Check which quadrants the branches occupy relative to the asymptotes.",
            "Use points one and two units from the vertical asymptote to estimate the stretch.",
        ],
        "sin": [
            "Find the midline first. That gives the vertical shift.",
            "Measure the distance from the midline to a peak or trough to get the amplitude.",
            "Locate where the curve crosses the midline going upward to identify the horizontal shift.",
        ],
        "cos": [
            "Find the midline first. That gives the vertical shift.",
            "Measure the distance from the midline to a peak or trough to get the amplitude.",
            "Use the horizontal distance for one full cycle to determine the period.",
        ],
    }
    return hints.get(problem.family_id, [])


def render_hint_deck(problem: ProblemInstance) -> None:
    if not st.session_state.tf_available_hints:
        st.session_state.tf_available_hints = build_hints(problem)

    hints = st.session_state.tf_available_hints
    shown = st.session_state.tf_hints_shown
    view_i = st.session_state.tf_hint_view_index

    st.markdown(f"Hint deck: **{len(shown)}** used")

    nav1, nav2, nav3 = st.columns([1, 3, 1])
    with nav1:
        prev_disabled = len(shown) == 0 or view_i <= 0
        if st.button("◀", key="tf_hint_prev", disabled=prev_disabled, width="stretch"):
            st.session_state.tf_hint_view_index = max(0, view_i - 1)
            st.rerun()

    with nav2:
        if not shown:
            st.caption(f"Hint 0 of {len(hints)}")
        else:
            st.caption(f"Hint {view_i + 1} of {len(shown)}")

    with nav3:
        next_disabled = len(hints) == 0 or (len(shown) >= len(hints) and view_i >= len(shown) - 1)
        if st.button("▶", key="tf_hint_next", disabled=next_disabled, width="stretch"):
            if len(shown) < len(hints):
                st.session_state.tf_hints_shown.append(hints[len(shown)])
                st.session_state.tf_hint_view_index = len(st.session_state.tf_hints_shown) - 1
            else:
                st.session_state.tf_hint_view_index = min(len(shown) - 1, view_i + 1)
            st.rerun()

    if not shown:
        st.markdown(
            """
            <div style="
                background: rgba(219, 234, 254, 0.35);
                border: 1px dashed rgba(147, 197, 253, 0.9);
                border-radius: 14px;
                padding: 14px 16px;
                color: #4b5563;
                line-height: 1.55;
                min-height: 120px;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
            ">
                <div style="font-size: 1rem;">Click ▶ to reveal the first hint.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif 0 <= view_i < len(shown):
        st.markdown(
            f"""
            <div style="
                background: rgba(219, 234, 254, 0.78);
                border: 1px solid rgba(147, 197, 253, 0.95);
                border-radius: 14px;
                padding: 14px 16px;
                color: #0f4c81;
                line-height: 1.55;
                min-height: 120px;
                display: flex;
                align-items: flex-start;
            ">
                <div style="font-size: 1.05rem;">{shown[view_i]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def check_params(
    problem: ProblemInstance,
    a_guess: float,
    h_guess: float,
    k_guess: float,
    b_guess: float | None = None,
) -> Tuple[bool, List[str]]:
    feedback = []
    ok = True

    if math.isclose(a_guess, problem.a, abs_tol=1e-9):
        feedback.append("Vertical scale/reflection: correct")
    else:
        ok = False
        if math.isclose(abs(a_guess), abs(problem.a), abs_tol=1e-9) and math.isclose(a_guess, -problem.a, abs_tol=1e-9):
            feedback.append("The size is right, but the reflection sign is wrong.")
        elif math.isclose(a_guess, -problem.a, abs_tol=1e-9):
            feedback.append("The graph has the opposite reflection from your choice.")
        else:
            feedback.append("The vertical scale/reflection value is not correct.")

    if math.isclose(h_guess, problem.h, abs_tol=1e-9):
        feedback.append("Horizontal shift: correct")
    else:
        ok = False
        if math.isclose(h_guess, -problem.h, abs_tol=1e-9):
            feedback.append("The horizontal shift sign is reversed.")
        else:
            feedback.append("Horizontal shift: not correct")

    if math.isclose(k_guess, problem.k, abs_tol=1e-9):
        feedback.append("Vertical shift: correct")
    else:
        ok = False
        if math.isclose(k_guess, -problem.k, abs_tol=1e-9):
            feedback.append("The vertical shift sign is reversed.")
        else:
            feedback.append("Vertical shift: not correct")

    if problem.b is not None:
        if b_guess is not None and math.isclose(float(b_guess), float(problem.b), abs_tol=1e-9):
            feedback.append("b value: correct")
        else:
            ok = False
            feedback.append("The b value is not correct.")

    return ok, feedback

def normalize_equation_text(s: str) -> str:
    s = s.lower().replace(" ", "")
    s = s.replace("\\pi", "pi").replace("π", "pi")
    s = s.replace("^2", "**2").replace("^3", "**3")
    s = s.replace("²", "**2").replace("³", "**3")
    return s

def insert_implicit_multiplication(expr: str) -> str:
    expr = re.sub(r"(?<=\d)(?=\()", "*", expr)
    expr = re.sub(r"(?<=\d)(?=x|pi|sin|cos|sqrt|abs)", "*", expr)
    expr = re.sub(r"(?<=x)(?=\()", "*", expr)
    expr = re.sub(r"(?<=pi)(?=\()", "*", expr)
    expr = re.sub(r"(?<=\))(?=\()", "*", expr)
    expr = re.sub(r"(?<=\))(?=x|pi|sin|cos|sqrt|abs|\d)", "*", expr)
    return expr


def convert_absolute_bars(expr: str) -> str:
    while "|" in expr:
        start = expr.find("|")
        end = expr.find("|", start + 1)
        if end == -1:
            raise ValueError("unmatched absolute value bar")
        expr = f"{expr[:start]}abs({expr[start + 1:end]}){expr[end + 1:]}"
    return expr


def parse_pi_expression(expr: str) -> float:
    prepared = normalize_equation_text(expr)
    prepared = insert_implicit_multiplication(prepared)
    if not prepared:
        raise ValueError("empty")
    if not re.fullmatch(r"[0-9pi+\-*/().]+", prepared):
        raise ValueError("invalid")
    prepared = prepared.replace("pi", "math.pi")
    return float(eval(prepared, {"__builtins__": {}}, {"math": math}))


def evaluate_student_expression(expr: str, x_values: np.ndarray) -> np.ndarray:
    prepared = normalize_equation_text(expr)
    if prepared.startswith("y="):
        prepared = prepared[2:]

    prepared = convert_absolute_bars(prepared)
    prepared = insert_implicit_multiplication(prepared)
    if "__" in prepared:
        raise ValueError("invalid expression")
    identifiers = re.findall(r"[a-z_]+", prepared)
    allowed_names = {"x", "abs", "sin", "cos", "sqrt", "pi"}
    if any(name not in allowed_names for name in identifiers):
        raise ValueError("invalid expression")
    allowed = {
        "x": x_values,
        "abs": np.abs,
        "sin": np.sin,
        "cos": np.cos,
        "sqrt": np.sqrt,
        "pi": np.pi,
    }
    return eval(prepared, {"__builtins__": {"__import__": __import__}}, allowed)


def acceptable_equation_forms(problem: ProblemInstance) -> List[str]:
    # Conservative first pass. Later this should use symbolic equivalence.
    target = normalize_equation_text(problem.display_equation)
    forms = {target}

    if problem.family_id == "quadratic":
        # Accept x^2 notation variant.
        forms.add(target.replace("²", "^2"))
    elif problem.family_id == "cubic":
        forms.add(target.replace("³", "^3"))

    return list(forms)


def check_equation(problem: ProblemInstance, student_equation: str) -> Tuple[bool, str]:
    try:
        xs = np.linspace(problem.x_min, problem.x_max, 800)
        true_y = evaluate_family(problem.family_id, xs, problem.a, problem.h, problem.k, problem.b)
        student_y = evaluate_student_expression(student_equation, xs)

        valid = np.isfinite(true_y) & np.isfinite(student_y)
        if not np.any(valid):
            return False, "I could not evaluate that equation on the graph window."

        if np.allclose(true_y[valid], student_y[valid], atol=1e-4, rtol=1e-4):
            return True, "Equation is correct."

        return False, "That equation is not equivalent to the graph."

    except Exception:
        return False, "I could not parse that equation. Try a form like y = 2cos(x - pi/2) + 1."

# -----------------------------------------------------------------------------
# Problem controls
# -----------------------------------------------------------------------------

def start_new_problem() -> None:
    st.session_state.tf_problem = generate_problem()
    st.session_state.tf_points = []
    st.session_state.tf_point_text = ""
    st.session_state.tf_point_label = "P"
    st.session_state.tf_point_error = ""
    st.session_state.tf_parent_choice = None
    st.session_state.tf_accepted_parent = None
    st.session_state.tf_a_guess = 1
    st.session_state.tf_h_guess = 0
    st.session_state.tf_h_guess_text = "0"
    st.session_state.tf_k_guess = 0
    st.session_state.tf_b_guess = None
    st.session_state.tf_equation_input = ""
    st.session_state.tf_available_hints = []
    st.session_state.tf_hints_shown = []
    st.session_state.tf_hint_view_index = -1
    st.session_state.tf_parent_submitted = False
    st.session_state.tf_parent_correct = False
    st.session_state.tf_params_submitted = False
    st.session_state.tf_params_correct = False
    st.session_state.tf_equation_submitted = False
    st.session_state.tf_equation_correct = False
    st.session_state.tf_feedback = []


def append_equation_text(snippet: str) -> None:
    st.session_state.tf_equation_input = f"{st.session_state.tf_equation_input}{snippet}"


def submit_equation_check(problem: ProblemInstance) -> None:
    ok, msg = check_equation(problem, st.session_state.tf_equation_input)
    st.session_state.tf_equation_submitted = True
    st.session_state.tf_equation_correct = ok
    st.session_state.tf_feedback = [msg]


# -----------------------------------------------------------------------------
# UI sections
# -----------------------------------------------------------------------------

def render_header(problem: ProblemInstance) -> None:
    c1, c2, c_gap, c3 = st.columns([3.6, 1.3, 1.3, 3.1], gap="small")
    with c1:
        st.markdown('<div class="tf-title">Function Transformations</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="tf-subtitle">Study the graph, identify the transformation form, describe the transformations, and build the equation.</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown('<div class="tf-header-actions">', unsafe_allow_html=True)
        if st.button("New graph", width="stretch"):
            start_new_problem()
            st.rerun()
        if st.button("Reveal answer", width="stretch"):
            b_text = f", b = {format_numeric_value(problem.b)}" if getattr(problem, "b", None) is not None else ""
            st.info(
                f"Parent: {problem.parent_label} | a = {format_numeric_value(problem.a)}, "
                f"h = {format_shift_value(problem.h)}, "
                f"k = {format_numeric_value(problem.k)}{b_text} | {problem.display_equation}"
            )
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        render_hint_deck(problem)

def render_teacher_debug(problem: ProblemInstance) -> None:
    username = str(st.session_state.get("username", "")).strip().lower()
    if username != "jim":
        return
    with st.expander("Teacher debug", expanded=False):
        st.write({
            "family": problem.family_id,
            "a": problem.a,
            "h": problem.h,
            "k": problem.k,
            "b": format_numeric_value(problem.b) if problem.b is not None else None,
            "angle_unit": problem.angle_unit,
            "equation": problem.display_equation,
            "key_points": [format_point(x, y) + f" [{label}]" for x, y, label in problem.key_points],
        })

def render_graph(problem: ProblemInstance) -> None:
    st.markdown('<div class="tf-graph">', unsafe_allow_html=True)
    st.markdown(
        '<div class="tf-small">You can zoom, pan, and reset the plot. Add points you notice on the graph below.</div>',
        unsafe_allow_html=True,
    )

    current_points = st.session_state.tf_points
    controls_col, graph_col = st.columns([1.2, 3.6])
    with controls_col:
        tab_add, tab_points, tab_step1, tab_step2 = st.tabs(
            ["Add point", "Points", "Step 1", "Step 2"]
        )

        with tab_add:
            c1, c2 = st.columns([1.4, 1])
            with c1:
                st.text_input("Point", key="tf_point_text", placeholder="(x,y)", autocomplete=None)
            with c2:
                st.text_input("Label", key="tf_point_label", autocomplete=None)

            st.button("Add point", width="stretch", on_click=add_point_from_inputs)
            st.button("Clear points", width="stretch", on_click=clear_points)
            if st.session_state.tf_point_error:
                st.error(st.session_state.tf_point_error)

        with tab_points:
            if current_points:
                df = pd.DataFrame(current_points)
                edited_df = st.data_editor(
                    df,
                    key="tf_points_editor",
                    width="stretch",
                    hide_index=True,
                    num_rows="dynamic",
                )
                current_points = normalize_points_rows(edited_df.to_dict("records"))
                if current_points != st.session_state.tf_points:
                    st.session_state.tf_points = current_points
            else:
                st.caption("No points added yet.")

        with tab_step1:
            render_parent_step(problem, wrapped=False)

        with tab_step2:
            render_parameter_step(problem, wrapped=False)

    with graph_col:
        fig = build_figure(problem, current_points)
        st.plotly_chart(fig, width="stretch")
        equation_col, button_col = st.columns([2, 1])
        with equation_col:
            render_equation_step(problem, wrapped=False, show_check_button=False)
        with button_col:
            st.button("Check equation", width="stretch", on_click=submit_equation_check, args=(problem,))
        if st.session_state.tf_equation_submitted:
            if st.session_state.tf_equation_correct:
                st.success(st.session_state.tf_feedback[0])
                st.balloons()
            else:
                st.error(st.session_state.tf_feedback[0])

    st.markdown('</div>', unsafe_allow_html=True)


def render_parent_step(problem: ProblemInstance, wrapped: bool = True) -> None:
    if wrapped:
        st.markdown('<div class="tf-card">', unsafe_allow_html=True)
    st.markdown("**Step 1: Choose the transformation form**")

    options = [
        ("quadratic", "Quadratic   (y = x²)"),
        ("absolute", "Absolute Value   (y = |x|)"),
        ("cubic", "Cubic   (y = x³)"),
        ("square_root", "Square Root"),
        ("reciprocal", "Reciprocal"),
        ("sin", "Sine   (y = sin(x))"),
        ("cos", "Cosine   (y = cos(x))"),
    ]

    selected = st.radio(
        "Choose the generic transformation equation",
        options=[o[0] for o in options],
        format_func=format_transformation_choice,
        key="tf_parent_choice",
        index=None,
    )

    if st.button("Check transformation form", width="stretch"):
        if selected is None:
            st.session_state.tf_parent_submitted = True
            st.session_state.tf_parent_correct = False
            st.session_state.tf_accepted_parent = None
            st.session_state.tf_feedback = ["Select a transformation form first."]
            st.rerun()
        ok, msg = check_parent(problem, selected)
        st.session_state.tf_parent_submitted = True
        st.session_state.tf_parent_correct = ok
        st.session_state.tf_accepted_parent = selected if ok else None
        st.session_state.tf_feedback = [msg]
        st.rerun()

    if st.session_state.tf_parent_submitted:
        msg = st.session_state.tf_feedback[0] if st.session_state.tf_feedback else ""
        if st.session_state.tf_parent_correct:
            st.success(msg)
        else:
            st.error(msg)

    if wrapped:
        st.markdown('</div>', unsafe_allow_html=True)


def render_parameter_step(problem: ProblemInstance, wrapped: bool = True) -> None:
    if wrapped:
        st.markdown('<div class="tf-card">', unsafe_allow_html=True)
    st.markdown("**Step 2: Describe the transformation**")
    if not st.session_state.tf_parent_correct:
        st.markdown(
            "Complete **Step 1** first. The correct generic equation will appear here with inputs for its transformation values."
        )
        st.info("Step 2 unlocks after the correct transformation form is selected.")
        if wrapped:
            st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown(
        "Enter the values that complete the selected transformation form.",
    )
    step_problem = get_effective_parent_problem(problem)
    has_b = step_problem.b is not None
    is_trig = step_problem.family_id in ("sin", "cos")
    st.markdown(f"**{step_problem.equation_template}**")
    a_guess = st.number_input("a", step=1.0, value=1.0, key="tf_a_guess")

    b_guess = None
    if has_b:
        b_values = [float(v) for v in step_problem.parameter_ranges.get("b", {}).get("values", [])]
        default_index = b_values.index(step_problem.b) if step_problem.b in b_values else 0
        b_guess = st.selectbox(
            "b",
            options=b_values,
            index=default_index,
            format_func=format_fraction_value,
            key="tf_b_guess",
        )

    if is_trig:
        h_values = [float(v) for v in step_problem.parameter_ranges.get("h", {}).get("values", [])]
        default_index = h_values.index(step_problem.h) if step_problem.h in h_values else 0
        h_guess = st.selectbox(
            "h",
            options=h_values,
            index=default_index,
            format_func=format_pi_text,
            key="tf_h_guess",
        )
    else:
        h_guess = st.number_input("h", step=0.5, value=0.0, key="tf_h_guess")
    k_guess = st.number_input("k", step=1.0, value=0.0, key="tf_k_guess")

    if st.button("Check transformation values", width="stretch"):
        ok, msgs = check_params(step_problem, float(a_guess), float(h_guess), float(k_guess), b_guess)
        st.session_state.tf_params_submitted = True
        st.session_state.tf_params_correct = ok
        st.session_state.tf_feedback = msgs
        st.rerun()

    if st.session_state.tf_params_submitted:
        if st.session_state.tf_params_correct:
            st.success("All transformation values are correct.")
        else:
            st.warning("Some values need revision.")
        for msg in st.session_state.tf_feedback:
            st.write(f"- {msg}")

    if wrapped:
        st.markdown('</div>', unsafe_allow_html=True)


def render_equation_step(problem: ProblemInstance, wrapped: bool = True, show_check_button: bool = True) -> None:
    if wrapped:
        st.markdown('<div class="tf-card">', unsafe_allow_html=True)

    label_col, input_col = st.columns([0.7, 2.3])
    with label_col:
        st.markdown("Equation of plot:")
    with input_col:
        st.text_input(
            "Equation of plot:",
            key="tf_equation_input",
            placeholder="Example: y = -2(x - 3)^2 + 1",
            autocomplete=None,
            label_visibility="collapsed",
        )
        insert_label_col, c1, c2, c3 = st.columns([0.8, 1, 1, 1])
        with insert_label_col:
            st.markdown("Insert")
        with c1:
            st.button("pi", width="stretch", on_click=append_equation_text, args=("pi",))
        with c2:
            st.button("| |", width="stretch", on_click=append_equation_text, args=("||",))
        with c3:
            st.button("sqrt()", width="stretch", on_click=append_equation_text, args=("sqrt()",))

    if show_check_button and st.button("Check equation", width="stretch"):
        submit_equation_check(problem)
        st.rerun()

    if wrapped:
        st.markdown('</div>', unsafe_allow_html=True)


def render_hint_block(problem: ProblemInstance) -> None:
    st.markdown('<div class="tf-card">', unsafe_allow_html=True)
    st.markdown("**Hint ideas**")

    hints = {
        "quadratic": [
            "Look for the vertex first.",
            "Does the parabola open up or down?",
            "Compare the width to y = x².",
        ],
        "absolute": [
            "Find the corner point first.",
            "Do the arms open up or down?",
            "Check how steep the arms are compared with y = |x|.",
        ],
        "cubic": [
            "Find the center turning point of the S-shape.",
            "Does the graph rise left-to-right or fall left-to-right?",
            "Compare nearby points one unit from the center.",
        ],
        "sin": [
            "Find the midline first.",
            "Measure the amplitude from the midline to a peak or trough.",
            "Track where one full cycle starts and repeats.",
        ],
        "cos": [
            "Find the midline first.",
            "Measure the amplitude from the midline to a peak or trough.",
            "Use the horizontal distance between consecutive peaks to determine the period.",
        ],
    }

    for hint in hints.get(problem.family_id, []):
        st.write(f"- {hint}")

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main render
# -----------------------------------------------------------------------------

def render_function_transformations_page() -> None:
    inject_transform_css()
    init_state()

    if st.session_state.tf_problem is None:
        start_new_problem()
    elif (
        not hasattr(st.session_state.tf_problem, "x_axis_display")
        or not hasattr(st.session_state.tf_problem, "b")
        or not hasattr(st.session_state.tf_problem, "parameter_ranges")
    ):
        start_new_problem()

    problem: ProblemInstance = st.session_state.tf_problem

    render_header(problem)

    render_graph(problem)
    # Remove this later if you do not want teacher visibility in TutorAssist.
    render_teacher_debug(problem)


# -----------------------------------------------------------------------------
# Standalone run hook
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    render_function_transformations_page()
