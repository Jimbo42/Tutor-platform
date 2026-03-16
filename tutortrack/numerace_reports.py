import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from shared.google_db import read_sheet_as_df

# -----------------------------
# Data prep helpers
# -----------------------------
def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def _to_bool(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.upper()
    return s.isin(["TRUE", "1", "YES", "Y"]).astype(bool)

def load_numerace_attempts_df() -> pd.DataFrame:
    df = read_sheet_as_df("numerace_attempts")
    if df.empty:
        return df

    # Ensure expected columns exist
    expected_cols = [
        "username", "domain", "skill", "subskill", "difficulty", "mastery_group",
        "question_id", "question_title", "correct", "missed",
        "question_seq", "attempts_on_question", "response_time", "choice_count"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""

    # normalize booleans
    df["correct"] = _to_bool(df["correct"])
    df["missed"] = _to_bool(df["missed"])

    # normalize numeric columns
    numeric_cols = ["question_seq", "attempts_on_question", "response_time", "choice_count"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # normalize text columns
    text_cols = [
        "username", "domain", "skill", "subskill", "difficulty",
        "mastery_group", "question_id", "question_title"
    ]
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # derived incorrect flag
    df["incorrect"] = (~df["correct"]) & (~df["missed"])

    # derived feedback_type
    df["feedback_type"] = "incorrect"
    df.loc[df["missed"], "feedback_type"] = "timeout"
    df.loc[(df["correct"]) & (df["attempts_on_question"].fillna(0) <= 1), "feedback_type"] = "correct_first_try"
    df.loc[(df["correct"]) & (df["attempts_on_question"].fillna(0) > 1), "feedback_type"] = "correct_after_retry"

    return df

def build_summary(
    df_attempts: pd.DataFrame,
    username: str,
    domain_filter: str = "All",
    min_questions: int = 1,
    summary_mode: str = "Subskill summary",
) -> pd.DataFrame:
    if df_attempts.empty:
        return pd.DataFrame()

    df = df_attempts.copy()

    # Student filter
    if username != "All students":
        df = df[df["username"].str.lower() == username.strip().lower()].copy()

    if df.empty:
        return pd.DataFrame()

    # Domain filter
    if domain_filter != "All":
        df = df[df["domain"] == domain_filter].copy()

    if df.empty:
        return pd.DataFrame()

    if summary_mode == "Question summary":
        group_cols = ["domain", "skill", "subskill", "question_id", "question_title"]
    else:
        group_cols = ["domain", "skill", "subskill"]

    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            students_seen=("username", "nunique"),
            questions_seen=("question_id", "count"),
            correct_count=("correct", "sum"),
            missed_count=("missed", "sum"),
            incorrect_count=("incorrect", "sum"),
            avg_response_time=("response_time", "mean"),
            avg_attempts=("attempts_on_question", "mean"),
            first_try_correct=("feedback_type", lambda s: (s == "correct_first_try").sum()),
        )
        .reset_index()
    )

    grouped["accuracy"] = grouped["correct_count"] / grouped["questions_seen"]
    grouped["missed_rate"] = grouped["missed_count"] / grouped["questions_seen"]
    grouped["first_try_rate"] = grouped["first_try_correct"] / grouped["questions_seen"]

    grouped["avg_response_time"] = grouped["avg_response_time"].round(2)
    grouped["avg_attempts"] = grouped["avg_attempts"].round(2)
    grouped["accuracy"] = grouped["accuracy"].round(3)
    grouped["missed_rate"] = grouped["missed_rate"].round(3)
    grouped["first_try_rate"] = grouped["first_try_rate"].round(3)

    grouped = grouped[grouped["questions_seen"] >= int(min_questions)].copy()

    grouped = grouped.sort_values(
        ["accuracy", "questions_seen", "domain", "skill", "subskill"],
        ascending=[True, False, True, True, True]
    ).reset_index(drop=True)

    return grouped

# -----------------------------
# Render helpers
# -----------------------------
def render_heat_table(df: pd.DataFrame, metric=None, summary_mode=None):
    if df is None or df.empty:
        st.info("No data available.")
        return

    view = df.copy()

    # --- preferred column order if present
    preferred = [
        "domain",
        "skill",
        "subskill",
        "students_seen",
        "questions_seen",
        "accuracy",
        "first_try_rate",
        "miss_rate",
        "avg_response_time",
        "attempts_per_question",
    ]
    cols = [c for c in preferred if c in view.columns] + [c for c in view.columns if c not in preferred]
    view = view[cols]

    # Rename for display
    view = view.rename(columns={
        "students_seen": "Students",
        "questions_seen": "Questions",
        "accuracy": "Accuracy",
        "first_try_rate": "First Try",
        "miss_rate": "Miss Rate",
        "avg_response_time": "Avg Time",
        "attempts_per_question": "Attempts/Q",
        "domain": "Domain",
        "skill": "Skill",
        "subskill": "Subskill",
    })

    # --- formatting columns AFTER rename
    int_cols = [c for c in ["Students", "Questions"] if c in view.columns]
    pct_cols = [c for c in ["Accuracy", "First Try", "Miss Rate"] if c in view.columns]
    dec_cols = [c for c in ["Avg Time", "Attempts/Q"] if c in view.columns]
    text_cols = [c for c in ["Domain", "Skill", "Subskill"] if c in view.columns]

    # format display
    fmt = {}
    for c in int_cols:
        fmt[c] = "{:.0f}"
    for c in pct_cols:
        fmt[c] = "{:.1%}"
    for c in dec_cols:
        fmt[c] = "{:.2f}"

    styler = (
        view.style
        .format(fmt, na_rep="—")
        .hide(axis="index")
        .set_properties(
            subset=text_cols,
            **{
                "text-align": "left",
                "white-space": "normal",
            }
        )
        .set_properties(
            subset=[c for c in view.columns if c not in text_cols],
            **{
                "text-align": "right",
            }
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
                "props": [
                    ("background-color", "#fafafa"),
                ],
            },
            {
                "selector": "tbody tr:hover",
                "props": [
                    ("background-color", "#f5faff"),
                ],
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

    # Good when higher
    for c in ["Accuracy", "First Try"]:
        if c in view.columns:
            styler = styler.background_gradient(
                subset=[c],
                cmap="RdYlGn",
                vmin=0.0,
                vmax=1.0,
            )

    # Bad when higher
    if "Miss Rate" in view.columns:
        styler = styler.background_gradient(
            subset=["Miss Rate"],
            cmap="RdYlGn_r",
            vmin=0.0,
            vmax=1.0,
        )

    st.dataframe(
        styler,
        width="stretch",
        height=min(480, 44 + 35 * len(view)),
    )

def render_weakest_bar(df_summary: pd.DataFrame, username: str, summary_mode: str, min_questions: int = 2):
    if df_summary.empty:
        st.info("No data available for chart.")
        return

    df = df_summary[df_summary["questions_seen"] >= min_questions].copy()
    if df.empty:
        st.info("Not enough data for chart yet.")
        return

    if summary_mode == "Question summary":
        label_col = "question_title" if "question_title" in df.columns else "question_id"
        caption = (
            "Lowest-accuracy question types across all students"
            if username == "All students"
            else f"Lowest-accuracy question types for {username}"
        )
    else:
        label_col = "subskill"
        caption = (
            "Lowest-accuracy subskills across all students"
            if username == "All students"
            else f"Lowest-accuracy subskills for {username}"
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
        xaxis=dict(
            range=[0, 1.08],  # creates the margin past 1.0
            tickformat=".0%",
            title="Accuracy",
        ),
        yaxis=dict(
            title="",
            automargin=True,
        ),
        margin=dict(
            l=220,  # gives labels room
            r=40,
            t=60,
            b=40,
        ),
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
        label_col = "question_title" if "question_title" in df.columns else "question_id"
        caption = (
            "Slowest question types across all students"
            if username == "All students"
            else f"Slowest question types for {username}"
        )
    else:
        label_col = "subskill"
        caption = (
            "Slowest subskills across all students"
            if username == "All students"
            else f"Slowest subskills for {username}"
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
        xaxis=dict(
            title="Average Response Time (seconds)",
            automargin=True,
        ),
        yaxis=dict(
            title="",
            automargin=True,
        ),
        margin=dict(
            l=220,
            r=40,
            t=60,
            b=40,
        ),
    )

    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, width="stretch")

def render_skill_radar(df_summary: pd.DataFrame, summary_mode: str, max_skills: int = 8, min_questions: int = 2):
    if summary_mode != "Subskill summary":
        st.info("Radar view is available for Subskill summary only.")
        return

    if df_summary.empty:
        st.info("No data available for radar chart.")
        return

    df = df_summary.copy()
    df = df[df["questions_seen"] >= min_questions].copy()

    if df.empty:
        st.info("Not enough data for radar chart yet.")
        return

    df = df.sort_values(["accuracy", "questions_seen", "subskill"], ascending=[True, False, True]).head(max_skills)

    df["radar_label"] = df.apply(
        lambda r: f"{r['subskill']} ({int(r['questions_seen'])})",
        axis=1
    )

    labels = df["radar_label"].tolist()
    values = df["accuracy"].tolist()

    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            name="Accuracy",
            hovertemplate="<b>%{theta}</b><br>Accuracy: %{r:.1%}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Skill Radar (weakest eligible subskills)",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat=".0%",
            )
        ),
        showlegend=False,
        height=560,
        margin=dict(l=60, r=60, t=70, b=60),
    )

    st.plotly_chart(fig, width="stretch")

def render_accuracy_speed_scatter(df_summary: pd.DataFrame, summary_mode: str):

    if df_summary.empty:
        st.info("No data available for scatter plot.")
        return

    if summary_mode == "Question summary":
        label_col = "question_title" if "question_title" in df_summary.columns else "question_id"
    else:
        label_col = "subskill"

    df = df_summary.copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["accuracy"],
            y=df["avg_response_time"],
            mode="markers",
            text=df[label_col],
            customdata=df["questions_seen"],
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
                "<br>Questions: %{customdata}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Accuracy vs Response Time",
        height=520,
        xaxis=dict(
            title="Accuracy",
            range=[0, 1.05],
            tickformat=".0%",
        ),
        yaxis=dict(
            title="Average Response Time (seconds)",
            automargin=True,
        ),
        margin=dict(
            l=70,
            r=40,
            t=60,
            b=60,
        ),
        shapes=[
            dict(type="line", x0=0.7, x1=0.7, y0=0, y1=df["avg_response_time"].max(),
                 line=dict(color="gray", dash="dash")),
            dict(type="line", x0=0, x1=1.05, y0=df["avg_response_time"].median(),
                 y1=df["avg_response_time"].median(),
                 line=dict(color="gray", dash="dash")),
        ],
    )

    st.plotly_chart(fig, width="stretch")

# -----------------------------
# Main page
# -----------------------------
def show_numerace_reports():
    st.title("NumeRace Reports")
    st.caption("Skill heatmap and summary reports from numerace_attempts")

    try:
        df_attempts = load_numerace_attempts_df()
    except Exception as e:
        st.error(f"Could not load NumeRace attempts: {e}")
        return

    if df_attempts.empty:
        st.warning("No NumeRace attempt data found in numerace_attempts.")
        return

    students = sorted([u for u in df_attempts["username"].dropna().unique().tolist() if str(u).strip()])
    student_options = ["All students"] + students
    domains = sorted([d for d in df_attempts["domain"].dropna().unique().tolist() if str(d).strip()])

    c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 2, 2])

    with c1:
        username = st.selectbox("Student", student_options, key="nr_report_student")

    with c2:
        domain_filter = st.selectbox("Domain", ["All"] + domains, key="nr_report_domain")

    with c3:
        min_questions = st.number_input("Min Q", min_value=1, max_value=100, value=1, step=1, key="nr_report_min_q")

    with c4:
        metric = st.selectbox(
            "Heat metric",
            ["accuracy", "first_try_rate", "missed_rate", "avg_response_time", "avg_attempts"],
            key="nr_report_metric",
        )

    with c5:
        summary_mode = st.selectbox(
            "Summary mode",
            ["Subskill summary", "Question summary"],
            key="nr_report_summary_mode",
        )

    df_summary = build_summary(
        df_attempts=df_attempts,
        username=username,
        domain_filter=domain_filter,
        min_questions=int(min_questions),
        summary_mode=summary_mode,
    )

    t1, t2, t3, t4 = st.tabs(["Heatmap", "Hardest", "Radar", "Scatter"])

    with t1:
        render_heat_table(df_summary, metric=metric, summary_mode=summary_mode)

    with t2:

        st.subheader("Lowest Accuracy")

        render_weakest_bar(
            df_summary,
            username=username,
            summary_mode=summary_mode,
            min_questions=max(2, int(min_questions))
        )

        st.markdown("---")

        st.subheader("Slowest Response Time")

        render_slowest_bar(
            df_summary,
            username=username,
            summary_mode=summary_mode,
            min_questions=max(2, int(min_questions))
        )

    with t3:
        render_skill_radar(
            df_summary,
            summary_mode=summary_mode,
            min_questions=max(2, int(min_questions))
        )

    with t4:

        render_accuracy_speed_scatter(
            df_summary,
            summary_mode=summary_mode
        )