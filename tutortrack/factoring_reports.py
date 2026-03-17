# factoring_reports.py
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from shared.google_db import read_sheet_as_df


LEVEL_LABELS = {
    1: "L1 Common Factor",
    2: "L2 Trinomial a=1",
    3: "L3 Trinomial a≠1",
    4: "L4 Diff Squares",
    5: "L5 Sum Squares",
    6: "L6 Vertex Form",
    7: "L7 Trinomial + GCF",
    8: "L8 Diff Squares + GCF",
    9: "L9 Perfect Square + GCF",
    10: "L10 Diff Cubes + GCF",
    11: "L11 Sum Cubes + GCF",
}


def _coerce_bool(s: pd.Series) -> pd.Series:
    return (
        s.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


def _coerce_num(s: pd.Series, default=0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _clean_text(s: pd.Series, default: str = "") -> pd.Series:
    return s.fillna(default).astype(str)


def load_factoring_attempts_df() -> pd.DataFrame:
    df = read_sheet_as_df("factoring_attempts")
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    expected_cols = [
        "timestamp", "attempt_id", "username", "round_key", "round_id",
        "question_seq", "level", "question_text", "target_expr", "input_text",
        "parsed_ok", "equivalent_to_target", "is_done", "is_progress_step",
        "invalid_step", "invalid_reason", "reactive_hint", "attempt_number",
        "response_time", "hints_used_so_far", "factor_tool_used_count",
        "steps_count", "current_expr_before", "current_expr_after"
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = ""

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["username"] = _clean_text(df["username"])
    df["round_key"] = _clean_text(df["round_key"])
    df["round_id"] = _clean_text(df["round_id"])
    df["question_seq"] = _coerce_num(df["question_seq"], 0).astype(int)
    df["level"] = _coerce_num(df["level"], 0).astype(int)
    df["question_text"] = _clean_text(df["question_text"])
    df["target_expr"] = _clean_text(df["target_expr"])
    df["input_text"] = _clean_text(df["input_text"])
    df["invalid_reason"] = _clean_text(df["invalid_reason"])
    df["reactive_hint"] = _clean_text(df["reactive_hint"])

    bool_cols = [
        "parsed_ok", "equivalent_to_target", "is_done", "is_progress_step", "invalid_step"
    ]
    for c in bool_cols:
        df[c] = _coerce_bool(df[c])

    num_cols = [
        "attempt_number", "response_time", "hints_used_so_far",
        "factor_tool_used_count", "steps_count"
    ]
    for c in num_cols:
        df[c] = _coerce_num(df[c], 0)

    df["level_label"] = df["level"].map(LEVEL_LABELS).fillna(df["level"].astype(str))
    df["student_label"] = df["username"].replace("", "Unknown")
    df = df.sort_values(["timestamp", "username", "round_key", "question_seq", "attempt_number", "attempt_id"])
    return df


def _apply_filters(df: pd.DataFrame, username: str, level_filter) -> pd.DataFrame:
    out = df.copy()
    if username != "All students":
        out = out[out["student_label"] == username]
    if level_filter != "All":
        out = out[out["level"] == int(level_filter)]
    return out


def _question_title_row(first: pd.Series) -> str:
    txt = str(first.get("question_text", "")).strip()
    if txt:
        return txt
    target = str(first.get("target_expr", "")).strip()
    if target:
        return target
    lvl = int(first.get("level", 0))
    seq = int(first.get("question_seq", 0))
    return f"Level {lvl} • Q{seq}"


def build_question_attempt_rollup(df_attempts: pd.DataFrame) -> pd.DataFrame:
    if df_attempts.empty:
        return pd.DataFrame()

    group_cols = ["username", "round_key", "question_seq"]

    rows = []
    for _, g in df_attempts.groupby(group_cols, dropna=False):
        g = g.sort_values(["timestamp", "attempt_number", "attempt_id"])
        first = g.iloc[0]
        last = g.iloc[-1]

        attempts_seen = len(g)
        correct = bool(g["is_done"].any())
        first_try = bool(correct and attempts_seen == 1)
        invalid_steps = int(g["invalid_step"].sum())
        progress_steps = int(g["is_progress_step"].sum())
        parse_errors = int((g["invalid_reason"] == "parse_error").sum())
        no_change_steps = int((g["invalid_reason"] == "no_change").sum())
        non_equivalent_steps = int((g["invalid_reason"] == "not_equivalent_to_target").sum())
        unproductive_equiv_steps = int((g["invalid_reason"] == "equivalent_but_not_progress").sum())

        row = {
            "username": first["student_label"],
            "round_key": first["round_key"],
            "round_id": first["round_id"],
            "question_seq": int(first["question_seq"]),
            "level": int(first["level"]),
            "level_label": first["level_label"],
            "question_title": _question_title_row(first),
            "target_expr": first["target_expr"],
            "questions_seen": 1,
            "correct": int(correct),
            "accuracy": float(correct),
            "first_try": int(first_try),
            "first_try_rate": float(first_try),
            "missed": int(not correct),
            "missed_rate": float(not correct),
            "attempts": attempts_seen,
            "avg_attempts": float(attempts_seen),
            "response_time_total": float(g["response_time"].sum()),
            "avg_response_time": float(g["response_time"].mean()) if attempts_seen else 0.0,
            "hints_used": float(g["hints_used_so_far"].max()),
            "factor_tool_uses": float(g["factor_tool_used_count"].max()),
            "invalid_steps": invalid_steps,
            "progress_steps": progress_steps,
            "parse_errors": parse_errors,
            "no_change_steps": no_change_steps,
            "non_equivalent_steps": non_equivalent_steps,
            "unproductive_equiv_steps": unproductive_equiv_steps,
            "completed_at": last["timestamp"],
        }
        rows.append(row)

    df_q = pd.DataFrame(rows)
    df_q["attempts_before_success"] = np.where(df_q["correct"] == 1, df_q["attempts"] - 1, df_q["attempts"])
    return df_q.sort_values(["username", "round_key", "question_seq"])


def build_summary(
    df_attempts: pd.DataFrame,
    username: str,
    level_filter,
    min_questions: int,
    summary_mode: str,
) -> pd.DataFrame:
    if df_attempts.empty:
        return pd.DataFrame()

    df = _apply_filters(df_attempts, username, level_filter)
    if df.empty:
        return pd.DataFrame()

    df_q = build_question_attempt_rollup(df)
    if df_q.empty:
        return pd.DataFrame()

    if summary_mode == "Question summary":
        group_cols = ["question_title", "level", "level_label"]
        sort_cols = ["accuracy", "questions_seen", "question_title"]
    else:
        group_cols = ["level", "level_label"]
        sort_cols = ["accuracy", "questions_seen", "level"]

    agg = (
        df_q.groupby(group_cols, dropna=False)
        .agg(
            questions_seen=("questions_seen", "sum"),
            correct=("correct", "sum"),
            missed=("missed", "sum"),
            first_try=("first_try", "sum"),
            accuracy=("accuracy", "mean"),
            first_try_rate=("first_try_rate", "mean"),
            missed_rate=("missed_rate", "mean"),
            avg_attempts=("avg_attempts", "mean"),
            avg_response_time=("avg_response_time", "mean"),
            hints_used=("hints_used", "mean"),
            factor_tool_uses=("factor_tool_uses", "mean"),
            invalid_steps=("invalid_steps", "mean"),
            progress_steps=("progress_steps", "mean"),
            parse_errors=("parse_errors", "mean"),
            no_change_steps=("no_change_steps", "mean"),
            non_equivalent_steps=("non_equivalent_steps", "mean"),
            unproductive_equiv_steps=("unproductive_equiv_steps", "mean"),
        )
        .reset_index()
    )

    if summary_mode == "Question summary":
        agg["subskill"] = agg["level_label"]
    else:
        agg["subskill"] = agg["level_label"]

    agg = agg[agg["questions_seen"] >= int(min_questions)].copy()
    if agg.empty:
        return agg

    agg = agg.sort_values(sort_cols, ascending=[True, False, True][:len(sort_cols)])
    return agg


def render_heat_table(df_summary: pd.DataFrame, metric: str, summary_mode: str):
    if df_summary.empty:
        st.info("No data available for table.")
        return

    if summary_mode == "Question summary":
        label_col = "question_title"
    else:
        label_col = "level_label"

    view = df_summary.copy()
    view = view.rename(columns={
        label_col: "Item",
        "questions_seen": "Questions",
        "accuracy": "Accuracy",
        "first_try_rate": "First Try",
        "missed_rate": "Miss Rate",
        "avg_attempts": "Avg Attempts",
        "avg_response_time": "Avg Time",
        "hints_used": "Avg Hints",
        "factor_tool_uses": "Avg Factor Tool",
        "invalid_steps": "Invalid Steps",
        "parse_errors": "Parse Errors",
        "no_change_steps": "No-change Steps",
        "non_equivalent_steps": "Non-equivalent",
        "unproductive_equiv_steps": "Unproductive Eqv",
    })

    keep_cols = [
        "Item", "Questions", "Accuracy", "First Try", "Miss Rate",
        "Avg Attempts", "Avg Time", "Avg Hints", "Avg Factor Tool",
        "Invalid Steps", "Parse Errors", "No-change Steps",
        "Non-equivalent", "Unproductive Eqv",
    ]
    view = view[[c for c in keep_cols if c in view.columns]]

    styler = (
        view.style
        .format({
            "Accuracy": "{:.1%}",
            "First Try": "{:.1%}",
            "Miss Rate": "{:.1%}",
            "Avg Attempts": "{:.2f}",
            "Avg Time": "{:.2f}s",
            "Avg Hints": "{:.2f}",
            "Avg Factor Tool": "{:.2f}",
            "Invalid Steps": "{:.2f}",
            "Parse Errors": "{:.2f}",
            "No-change Steps": "{:.2f}",
            "Non-equivalent": "{:.2f}",
            "Unproductive Eqv": "{:.2f}",
        })
        .set_properties(
            subset=["Item"],
            **{
                "text-align": "left",
                "white-space": "normal",
            }
        )
        .set_properties(
            subset=[c for c in view.columns if c != "Item"],
            **{"text-align": "right"}
        )
        .set_table_styles([
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#f3f4f6"),
                    ("color", "#374151"),
                    ("font-weight", "600"),
                    ("border-bottom", "1px solid #d1d5db"),
                    ("padding", "8px 10px"),
                    ("text-align", "left"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("border-bottom", "1px solid #eef2f7"),
                    ("padding", "7px 10px"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#fafafa")],
            },
            {
                "selector": "tbody tr:hover",
                "props": [("background-color", "#f5faff")],
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("font-size", "0.95rem"),
                    ("width", "100%"),
                ],
            },
        ])
    )

    metric_to_col = {
        "accuracy": "Accuracy",
        "first_try_rate": "First Try",
        "missed_rate": "Miss Rate",
        "avg_response_time": "Avg Time",
        "avg_attempts": "Avg Attempts",
        "invalid_steps": "Invalid Steps",
        "hints_used": "Avg Hints",
        "factor_tool_uses": "Avg Factor Tool",
    }

    chosen = metric_to_col.get(metric)
    if chosen in ["Accuracy", "First Try"]:
        styler = styler.background_gradient(subset=[chosen], cmap="RdYlGn", vmin=0.0, vmax=1.0)
    elif chosen == "Miss Rate":
        styler = styler.background_gradient(subset=[chosen], cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
    elif chosen in ["Avg Time", "Avg Attempts", "Invalid Steps", "Avg Hints", "Avg Factor Tool"]:
        styler = styler.background_gradient(subset=[chosen], cmap="RdYlGn_r")

    st.dataframe(styler, width="stretch", height=min(520, 44 + 35 * len(view)))

def render_weakest_bar(df_summary: pd.DataFrame, username: str, summary_mode: str, min_questions: int = 2):
    if df_summary.empty:
        st.info("No data available for chart.")
        return

    df = df_summary[df_summary["questions_seen"] >= min_questions].copy()
    if df.empty:
        st.info("Not enough data for chart yet.")
        return

    if summary_mode == "Question summary":
        label_col = "question_title"
        caption = (
            "Lowest-accuracy question types across all students"
            if username == "All students"
            else f"Lowest-accuracy question types for {username}"
        )
    else:
        label_col = "level_label"
        caption = (
            "Lowest-accuracy factoring levels across all students"
            if username == "All students"
            else f"Lowest-accuracy factoring levels for {username}"
        )

    chart_df = (
        df[[label_col, "accuracy", "questions_seen"]]
        .sort_values("accuracy", ascending=True)
        .head(10)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart_df["accuracy"],
            y=chart_df[label_col],
            orientation="h",
            marker_color="#1f77b4",
            text=(chart_df["accuracy"] * 100).round(1).astype(str) + "%",
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Accuracy: %{x:.1%}<br>Questions: %{customdata}<extra></extra>",
            customdata=chart_df["questions_seen"],
        )
    )

    fig.update_layout(
        title=caption,
        height=450,
        xaxis=dict(range=[0, 1.08], tickformat=".0%", title="Accuracy"),
        yaxis=dict(title="", automargin=True),
        margin=dict(l=220, r=40, t=60, b=40),
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")

def render_slowest_bar(df_summary: pd.DataFrame, username: str, summary_mode: str, min_questions: int = 2):
    if df_summary.empty:
        st.info("No data available for chart.")
        return

    df = df_summary[df_summary["questions_seen"] >= min_questions].copy()
    if df.empty:
        st.info("Not enough data for chart yet.")
        return

    if summary_mode == "Question summary":
        label_col = "question_title"
        caption = (
            "Slowest question types across all students"
            if username == "All students"
            else f"Slowest question types for {username}"
        )
    else:
        label_col = "level_label"
        caption = (
            "Slowest factoring levels across all students"
            if username == "All students"
            else f"Slowest factoring levels for {username}"
        )

    chart_df = (
        df[[label_col, "avg_response_time", "questions_seen"]]
        .sort_values("avg_response_time", ascending=False)
        .head(10)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart_df["avg_response_time"],
            y=chart_df[label_col],
            orientation="h",
            marker_color="#e15759",
            text=chart_df["avg_response_time"].round(1).astype(str) + "s",
            textposition="outside",
            customdata=chart_df["questions_seen"],
            hovertemplate="<b>%{y}</b><br>Avg Time: %{x:.2f}s<br>Questions: %{customdata}<extra></extra>",
        )
    )

    fig.update_layout(
        title=caption,
        height=450,
        xaxis=dict(title="Average Response Time (seconds)", automargin=True),
        yaxis=dict(title="", automargin=True),
        margin=dict(l=220, r=40, t=60, b=40),
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")


def render_invalid_steps_bar(df_summary: pd.DataFrame, username: str, summary_mode: str, min_questions: int = 2):
    if df_summary.empty:
        st.info("No data available for chart.")
        return

    df = df_summary[df_summary["questions_seen"] >= min_questions].copy()
    if df.empty:
        st.info("Not enough data for chart yet.")
        return

    if summary_mode == "Question summary":
        label_col = "question_title"
        caption = (
            "Most invalid-step-heavy questions across all students"
            if username == "All students"
            else f"Most invalid-step-heavy questions for {username}"
        )
    else:
        label_col = "level_label"
        caption = (
            "Most invalid-step-heavy levels across all students"
            if username == "All students"
            else f"Most invalid-step-heavy levels for {username}"
        )

    chart_df = (
        df[[label_col, "invalid_steps", "questions_seen"]]
        .sort_values("invalid_steps", ascending=False)
        .head(10)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart_df["invalid_steps"],
            y=chart_df[label_col],
            orientation="h",
            marker_color="#f28e2b",
            text=chart_df["invalid_steps"].round(2).astype(str),
            textposition="outside",
            customdata=chart_df["questions_seen"],
            hovertemplate="<b>%{y}</b><br>Invalid steps/q: %{x:.2f}<br>Questions: %{customdata}<extra></extra>",
        )
    )

    fig.update_layout(
        title=caption,
        height=450,
        xaxis=dict(title="Average Invalid Steps per Question", automargin=True),
        yaxis=dict(title="", automargin=True),
        margin=dict(l=220, r=40, t=60, b=40),
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")

def render_level_effort_bubble(df_summary: pd.DataFrame, min_questions: int = 2):
    if df_summary.empty:
        st.info("No data available for level effort chart.")
        return

    if "level" not in df_summary.columns:
        st.info("This chart is available for Level summary only.")
        return

    df = df_summary[df_summary["questions_seen"] >= min_questions].copy()
    if df.empty:
        st.info("Not enough data for level effort chart yet.")
        return

    df = df.sort_values("level").copy()
    df["avg_steps_per_question"] = df["progress_steps"] + df["invalid_steps"]

    x_med = float(df["avg_response_time"].median()) if not df.empty else 0.0
    y_med = float(df["avg_steps_per_question"].median()) if not df.empty else 0.0

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["avg_response_time"],
            y=df["avg_steps_per_question"],
            mode="markers+text",
            text=df["level_label"],
            textposition="top center",
            customdata=np.stack(
                [
                    df["questions_seen"],
                    df["invalid_steps"],
                    df["progress_steps"],
                ],
                axis=1,
            ),
            marker=dict(
                size=df["questions_seen"] * 4 + 10,
                color=df["invalid_steps"],
                colorscale="YlOrRd",
                showscale=True,
                colorbar=dict(title="Invalid/q"),
                line=dict(width=1, color="black"),
                sizemode="diameter",
                opacity=0.85,
            ),
            hovertemplate=(
                "<b>%{text}</b>"
                "<br>Avg Time: %{x:.2f}s"
                "<br>Avg Steps/Q: %{y:.2f}"
                "<br>Questions: %{customdata[0]}"
                "<br>Invalid Steps/Q: %{customdata[1]:.2f}"
                "<br>Progress Steps/Q: %{customdata[2]:.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Level Effort: Time vs Steps",
        height=540,
        xaxis=dict(title="Average Response Time (seconds)", automargin=True),
        yaxis=dict(title="Average Submitted Steps per Question", automargin=True),
        margin=dict(l=70, r=40, t=60, b=60),
        shapes=[
            dict(
                type="line",
                x0=x_med, x1=x_med,
                y0=0, y1=max(df["avg_steps_per_question"]) * 1.1,
                line=dict(color="gray", dash="dash"),
            ),
            dict(
                type="line",
                x0=0, x1=max(df["avg_response_time"]) * 1.1,
                y0=y_med, y1=y_med,
                line=dict(color="gray", dash="dash"),
            ),
        ],
        annotations=[
            dict(x=x_med, y=max(df["avg_steps_per_question"]) * 1.08, text="slower →", showarrow=False),
            dict(x=max(df["avg_response_time"]) * 1.02, y=y_med, text="more steps ↑", showarrow=False),
        ],
    )

    st.plotly_chart(fig, width="stretch")

def render_accuracy_speed_scatter(df_summary: pd.DataFrame, summary_mode: str):
    if df_summary.empty:
        st.info("No data available for scatter plot.")
        return

    label_col = "question_title" if summary_mode == "Question summary" else "level_label"
    df = df_summary.copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["accuracy"],
            y=df["avg_response_time"],
            mode="markers",
            text=df[label_col],
            customdata=np.stack(
                [
                    df["questions_seen"],
                    df["invalid_steps"],
                    df["hints_used"],
                    df["factor_tool_uses"],
                ],
                axis=1,
            ),
            marker=dict(
                size=12,
                color=df["accuracy"],
                colorscale="RdYlGn",
                cmin=0,
                cmax=1,
                showscale=True,
                colorbar=dict(title="Accuracy"),
                line=dict(width=1, color="black"),
            ),
            hovertemplate=(
                "<b>%{text}</b>"
                "<br>Accuracy: %{x:.1%}"
                "<br>Avg Time: %{y:.2f}s"
                "<br>Questions: %{customdata[0]}"
                "<br>Invalid/q: %{customdata[1]:.2f}"
                "<br>Hints/q: %{customdata[2]:.2f}"
                "<br>Factor tool/q: %{customdata[3]:.2f}"
                "<extra></extra>"
            ),
        )
    )

    max_y = max(1.0, float(df["avg_response_time"].max())) if not df.empty else 1.0
    median_y = float(df["avg_response_time"].median()) if not df.empty else 0.0

    fig.update_layout(
        title="Accuracy vs Response Time",
        height=520,
        xaxis=dict(title="Accuracy", range=[0, 1.05], tickformat=".0%"),
        yaxis=dict(title="Average Response Time (seconds)", automargin=True),
        margin=dict(l=70, r=40, t=60, b=60),
        shapes=[
            dict(type="line", x0=0.7, x1=0.7, y0=0, y1=max_y, line=dict(color="gray", dash="dash")),
            dict(type="line", x0=0, x1=1.05, y0=median_y, y1=median_y, line=dict(color="gray", dash="dash")),
        ],
    )
    st.plotly_chart(fig, width="stretch")


def render_hints_success_scatter(df_summary: pd.DataFrame, summary_mode: str):
    if df_summary.empty:
        st.info("No data available for scatter plot.")
        return

    label_col = "question_title" if summary_mode == "Question summary" else "level_label"
    df = df_summary.copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["hints_used"],
            y=df["accuracy"],
            mode="markers",
            text=df[label_col],
            customdata=np.stack([df["questions_seen"], df["avg_attempts"], df["invalid_steps"]], axis=1),
            marker=dict(
                size=12,
                color=df["invalid_steps"],
                colorscale="YlOrRd",
                showscale=True,
                colorbar=dict(title="Invalid/q"),
                line=dict(width=1, color="black"),
            ),
            hovertemplate=(
                "<b>%{text}</b>"
                "<br>Hints/q: %{x:.2f}"
                "<br>Accuracy: %{y:.1%}"
                "<br>Questions: %{customdata[0]}"
                "<br>Avg Attempts: %{customdata[1]:.2f}"
                "<br>Invalid/q: %{customdata[2]:.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Hints Used vs Accuracy",
        height=520,
        xaxis=dict(title="Average Hints Used per Question"),
        yaxis=dict(title="Accuracy", range=[0, 1.05], tickformat=".0%"),
        margin=dict(l=70, r=40, t=60, b=60),
    )
    st.plotly_chart(fig, width="stretch")


def render_invalid_reason_heatmap(df_attempts: pd.DataFrame, username: str, level_filter):
    if df_attempts.empty:
        st.info("No data available for invalid-step heatmap.")
        return

    df = _apply_filters(df_attempts, username, level_filter).copy()
    df = df[df["invalid_step"]].copy()
    if df.empty:
        st.info("No invalid steps recorded yet.")
        return

    reasons = [
        "parse_error",
        "not_equivalent_to_target",
        "no_change",
        "equivalent_but_not_progress",
        "level5_wrong_text",
    ]

    pivot = (
        df.assign(reason=df["invalid_reason"].replace("", "other"))
        .pivot_table(
            index="level_label",
            columns="reason",
            values="attempt_id",
            aggfunc="count",
            fill_value=0,
        )
    )

    cols = [c for c in reasons if c in pivot.columns] + [c for c in pivot.columns if c not in reasons]
    pivot = pivot[cols]

    z = pivot.values
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="YlOrRd",
            hovertemplate="Level: %{y}<br>Reason: %{x}<br>Count: %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Invalid Step Types by Level",
        height=max(360, 100 + 36 * len(pivot.index)),
        xaxis=dict(title="Invalid Reason"),
        yaxis=dict(title="Level", automargin=True),
        margin=dict(l=140, r=30, t=60, b=60),
    )
    st.plotly_chart(fig, width="stretch")


def show_factoring_reports():
    st.title("Factoring Reports")
    st.caption("Analytics from factoring_attempts")

    try:
        df_attempts = load_factoring_attempts_df()
    except Exception as e:
        st.error(f"Could not load factoring attempts: {e}")
        return

    if df_attempts.empty:
        st.warning("No factoring attempt data found in factoring_attempts.")
        return

    students = sorted([u for u in df_attempts["student_label"].dropna().unique().tolist() if str(u).strip()])
    student_options = ["All students"] + students
    levels = sorted([int(x) for x in df_attempts["level"].dropna().unique().tolist() if int(x) > 0])

    c1, c2, c3, c4, c5 = st.columns([2, 1.4, 1, 1.6, 1.6])

    with c1:
        username = st.selectbox("Student", student_options, key="fp_report_student")

    with c2:
        level_filter = st.selectbox("Level", ["All"] + levels, key="fp_report_level")

    with c3:
        min_questions = st.number_input("Min Q", min_value=1, max_value=100, value=1, step=1, key="fp_report_min_q")

    with c4:
        metric = st.selectbox(
            "Heat metric",
            [
                "accuracy",
                "first_try_rate",
                "missed_rate",
                "avg_response_time",
                "avg_attempts",
                "invalid_steps",
                "hints_used",
                "factor_tool_uses",
            ],
            key="fp_report_metric",
        )

    with c5:
        summary_mode = st.selectbox(
            "Summary mode",
            ["Level summary", "Question summary"],
            key="fp_report_summary_mode",
        )

    df_summary = build_summary(
        df_attempts=df_attempts,
        username=username,
        level_filter=level_filter,
        min_questions=int(min_questions),
        summary_mode=summary_mode,
    )

    t1, t2, t3, t4, t5 = st.tabs(["Heatmap", "Hardest", "Level Effort", "Scatter", "Diagnostics"])

    with t1:
        render_heat_table(df_summary, metric=metric, summary_mode=summary_mode)

    with t2:
        st.subheader("Lowest Accuracy")
        render_weakest_bar(
            df_summary,
            username=username,
            summary_mode=summary_mode,
            min_questions=max(2, int(min_questions)),
        )

        st.markdown("---")
        st.subheader("Slowest Response Time")
        render_slowest_bar(
            df_summary,
            username=username,
            summary_mode=summary_mode,
            min_questions=max(2, int(min_questions)),
        )

        st.markdown("---")
        st.subheader("Most Invalid Steps")
        render_invalid_steps_bar(
            df_summary,
            username=username,
            summary_mode=summary_mode,
            min_questions=max(2, int(min_questions)),
        )

    with t3:
        if summary_mode != "Level summary":
            st.info("This tab is most useful in Level summary mode.")
        else:
            render_level_effort_bubble(
                df_summary,
                min_questions=max(2, int(min_questions)),
            )

    with t4:
        render_accuracy_speed_scatter(df_summary, summary_mode=summary_mode)
        st.markdown("---")
        render_hints_success_scatter(df_summary, summary_mode=summary_mode)

    with t5:
        render_invalid_reason_heatmap(
            df_attempts=df_attempts,
            username=username,
            level_filter=level_filter,
        )

        st.markdown("---")
        st.subheader("Raw summary")
        if df_summary.empty:
            st.info("No filtered summary rows.")
        else:
            st.dataframe(df_summary, width="stretch", height=min(520, 44 + 35 * len(df_summary)))

