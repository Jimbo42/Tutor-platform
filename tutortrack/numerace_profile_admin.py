import streamlit as st
from streamlit import session_state as ss
import pandas as pd

from shared.google_db import (
    read_sheet_as_df,
    get_sheet,
    find_row_number_by_values,
    upsert_row_by_headers,
)


PROFILE_TAB = "numerace_user_profile"

ACTION_OPTIONS = [
    "maintain",
    "stretch",
    "fluency_practice",
    "stabilize",
    "support",
]

def _action_chip(action: str) -> str:
    action = str(action or "").strip()

    cls_map = {
        "support": "nr-chip-support",
        "stabilize": "nr-chip-stabilize",
        "fluency_practice": "nr-chip-fluency",
        "maintain": "nr-chip-maintain",
        "stretch": "nr-chip-stretch",
    }
    cls = cls_map.get(action, "nr-chip-default")
    label = action.replace("_", " ").title() if action else "Unknown"
    return f"<span class='nr-chip {cls}'>{label}</span>"


def _multiplier_flag(mult: float) -> str:
    m = _safe_float(mult, 1.0)
    if m >= 1.45:
        return "🔴"
    if m >= 1.15:
        return "🟠"
    if m <= 0.80:
        return "🔵"
    return "🟢"

def _safe_float(v, default=0.0):
    try:
        if v in ("", None):
            return default
        return float(v)
    except Exception:
        return default


def _safe_int(v, default=0):
    try:
        if v in ("", None):
            return default
        return int(float(v))
    except Exception:
        return default


def _load_profiles_df() -> pd.DataFrame:
    df = read_sheet_as_df(PROFILE_TAB)
    if df.empty:
        return df

    numeric_cols = [
        "questions_seen",
        "correct_count",
        "missed_count",
        "incorrect_count",
        "accuracy",
        "avg_response_time",
        "recent_accuracy",
        "recent_avg_response_time",
        "current_multiplier",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["username", "domain", "skill", "subskill", "mastery_group", "recommended_action", "last_seen"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    return df


def _profile_match_dict(row: pd.Series) -> dict:
    return {
        "username": str(row.get("username", "")).strip(),
        "domain": str(row.get("domain", "")).strip(),
        "skill": str(row.get("skill", "")).strip(),
        "subskill": str(row.get("subskill", "")).strip(),
        "mastery_group": str(row.get("mastery_group", "")).strip(),
    }


def _delete_profile_row(match_dict: dict) -> bool:
    ws = get_sheet(PROFILE_TAB)
    row_num = find_row_number_by_values(PROFILE_TAB, match_dict)
    if row_num is None:
        return False
    ws.delete_rows(row_num)
    return True


def _delete_profiles_for_user(username: str) -> int:
    username = str(username or "").strip()
    if not username:
        return 0

    ws = get_sheet(PROFILE_TAB)
    all_values = ws.get_all_values()
    if len(all_values) <= 1:
        return 0

    headers = all_values[0]
    try:
        username_idx = headers.index("username")
    except ValueError:
        raise ValueError("Sheet numerace_user_profile is missing 'username' header.")

    rows_to_delete = []
    for row_num, row in enumerate(all_values[1:], start=2):
        val = row[username_idx] if username_idx < len(row) else ""
        if str(val).strip() == username:
            rows_to_delete.append(row_num)

    # delete bottom-up so row numbers do not shift
    for row_num in reversed(rows_to_delete):
        ws.delete_rows(row_num)

    return len(rows_to_delete)


def _reset_profile_row(row: pd.Series) -> None:
    match_dict = _profile_match_dict(row)

    reset_row = {
        "timestamp_updated": row.get("timestamp_updated", ""),
        "username": match_dict["username"],
        "domain": match_dict["domain"],
        "skill": match_dict["skill"],
        "subskill": match_dict["subskill"],
        "mastery_group": match_dict["mastery_group"],
        "questions_seen": 0,
        "correct_count": 0,
        "missed_count": 0,
        "incorrect_count": 0,
        "accuracy": 0.0,
        "avg_response_time": 0.0,
        "recent_accuracy": 0.0,
        "recent_avg_response_time": 0.0,
        "current_multiplier": 1.0,
        "recommended_action": "maintain",
        "last_seen": "",
    }

    upsert_row_by_headers(PROFILE_TAB, match_dict, reset_row)


def _update_profile_override(
    row: pd.Series,
    *,
    current_multiplier: float,
    recommended_action: str,
) -> None:
    match_dict = _profile_match_dict(row)

    updated = row.to_dict()
    updated["current_multiplier"] = float(current_multiplier)
    updated["recommended_action"] = str(recommended_action)

    upsert_row_by_headers(PROFILE_TAB, match_dict, updated)


def show_numerace_profile_admin():
    st.title("📘 NumeRace Profile Admin")

    st.markdown(
        """
        <style>
        .nr-chip {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .nr-chip-support {
            background: #fde2e1;
            color: #9b1c1c;
        }

        .nr-chip-stabilize {
            background: #ffe8cc;
            color: #9a4d00;
        }

        .nr-chip-fluency {
            background: #fff3cd;
            color: #7a5a00;
        }

        .nr-chip-maintain {
            background: #e6f4ea;
            color: #1e6b3a;
        }

        .nr-chip-stretch {
            background: #dbeafe;
            color: #1d4ed8;
        }

        .nr-chip-default {
            background: #e5e7eb;
            color: #374151;
        }

        .nr-admin-note {
            color: #6b7280;
            font-size: 0.92rem;
            margin-top: -0.25rem;
            margin-bottom: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔄 Refresh profiles", width="stretch"):
        st.rerun()

    df = _load_profiles_df()

    if df.empty:
        st.info("No NumeRace profile rows found.")
        return

    st.markdown("### Filters")
    c1, c2, c3, c4, c5 = st.columns(5)

    usernames = ["All"] + sorted([x for x in df["username"].dropna().unique().tolist() if str(x).strip()])
    domains = ["All"] + sorted([x for x in df["domain"].dropna().unique().tolist() if str(x).strip()])
    skills = ["All"] + sorted([x for x in df["skill"].dropna().unique().tolist() if str(x).strip()])
    mastery_groups = ["All"] + sorted([x for x in df["mastery_group"].dropna().unique().tolist() if str(x).strip()])
    actions = ["All"] + sorted([x for x in df["recommended_action"].dropna().unique().tolist() if str(x).strip()])

    with c1:
        f_user = st.selectbox("Student", usernames, key="nrp_f_user")
    with c2:
        f_domain = st.selectbox("Domain", domains, key="nrp_f_domain")
    with c3:
        f_skill = st.selectbox("Skill", skills, key="nrp_f_skill")
    with c4:
        f_mg = st.selectbox("Mastery group", mastery_groups, key="nrp_f_mg")
    with c5:
        f_action = st.selectbox("Action", actions, key="nrp_f_action")

    c6, c7 = st.columns(2)
    with c6:
        only_nonmaintain = st.checkbox("Show only non-maintain", key="nrp_nonmaintain")
    with c7:
        only_low_acc = st.checkbox("Show only recent accuracy < 0.80", key="nrp_lowacc")

    filtered = df.copy()

    if f_user != "All":
        filtered = filtered[filtered["username"] == f_user]
    if f_domain != "All":
        filtered = filtered[filtered["domain"] == f_domain]
    if f_skill != "All":
        filtered = filtered[filtered["skill"] == f_skill]
    if f_mg != "All":
        filtered = filtered[filtered["mastery_group"] == f_mg]
    if f_action != "All":
        filtered = filtered[filtered["recommended_action"] == f_action]
    if only_nonmaintain:
        filtered = filtered[filtered["recommended_action"] != "maintain"]
    if only_low_acc:
        filtered = filtered[filtered["recent_accuracy"].fillna(0) < 0.80]

    st.markdown("### Summary")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Profile rows", len(filtered))
    with s2:
        support_count = int((filtered["recommended_action"] == "support").sum()) if "recommended_action" in filtered.columns else 0
        st.metric("Support", support_count)
    with s3:
        stretch_count = int((filtered["recommended_action"] == "stretch").sum()) if "recommended_action" in filtered.columns else 0
        st.metric("Stretch", stretch_count)
    with s4:
        avg_mult = filtered["current_multiplier"].mean() if "current_multiplier" in filtered.columns and len(filtered) else 0.0
        st.metric("Avg multiplier", f"{avg_mult:.2f}")

    st.markdown("### Profile table")
    st.markdown(
        "<div class='nr-admin-note'>The colored status chip reflects the current adaptive recommendation for each profile row.</div>",
        unsafe_allow_html=True,
    )

    display_cols = [
        "username",
        "recommended_action",
        "domain",
        "skill",
        "subskill",
        "mastery_group",
        "questions_seen",
        "accuracy",
        "avg_response_time",
        "recent_accuracy",
        "recent_avg_response_time",
        "current_multiplier",
        "last_seen",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    styled = filtered.copy()

    styled["status"] = styled["recommended_action"].apply(_action_chip)
    styled["mult"] = styled["current_multiplier"].apply(_multiplier_flag) + " " + styled["current_multiplier"].fillna(1.0).map(lambda x: f"{float(x):.2f}")

    styled_view_cols = [
        "username",
        "status",
        "domain",
        "skill",
        "subskill",
        "mastery_group",
        "questions_seen",
        "accuracy",
        "recent_accuracy",
        "avg_response_time",
        "recent_avg_response_time",
        "mult",
        "last_seen",
    ]
    styled_view_cols = [c for c in styled_view_cols if c in styled.columns]

    st.markdown(
        styled[styled_view_cols].to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )

    with st.expander("Show raw dataframe"):
        st.dataframe(
            filtered[display_cols],
            width="stretch",
            hide_index=False,
        )

    if filtered.empty:
        st.info("No rows match the current filters.")
        return

    st.markdown("### Row actions")

    row_options = list(filtered.index)
    selected_idx = st.selectbox(
        "Select profile row",
        row_options,
        format_func=lambda idx: (
            f"{filtered.loc[idx, 'username']} | "
            f"{filtered.loc[idx, 'domain']} / "
            f"{filtered.loc[idx, 'skill']} / "
            f"{filtered.loc[idx, 'subskill']} / "
            f"{filtered.loc[idx, 'mastery_group']}"
        ),
        key="nrp_selected_idx",
    )

    row = filtered.loc[selected_idx]

    d1, d2 = st.columns([1.2, 1.8])
    with d1:
        st.markdown("#### Current values")
        st.write(f"**User:** {row.get('username', '')}")
        st.write(f"**Domain:** {row.get('domain', '')}")
        st.write(f"**Skill:** {row.get('skill', '')}")
        st.write(f"**Subskill:** {row.get('subskill', '')}")
        st.write(f"**Mastery group:** {row.get('mastery_group', '')}")
        st.write(f"**Questions seen:** {_safe_int(row.get('questions_seen', 0))}")
        st.write(f"**Accuracy:** {_safe_float(row.get('accuracy', 0.0)):.3f}")
        st.write(f"**Recent accuracy:** {_safe_float(row.get('recent_accuracy', 0.0)):.3f}")
        st.write(f"**Current multiplier:** {_safe_float(row.get('current_multiplier', 1.0)):.2f}")
        st.markdown(
            f"**Recommended action:** {_action_chip(row.get('recommended_action', ''))}",
            unsafe_allow_html=True,
        )
    with d2:
        st.markdown("#### Override / maintenance")

        new_multiplier = st.number_input(
            "Current multiplier",
            min_value=0.10,
            max_value=3.00,
            value=float(_safe_float(row.get("current_multiplier", 1.0), 1.0)),
            step=0.05,
            key="nrp_edit_multiplier",
        )

        current_action = str(row.get("recommended_action", "maintain")).strip() or "maintain"
        if current_action not in ACTION_OPTIONS:
            current_action = "maintain"

        new_action = st.selectbox(
            "Recommended action",
            ACTION_OPTIONS,
            index=ACTION_OPTIONS.index(current_action),
            key="nrp_edit_action",
        )

        a1, a2, a3 = st.columns(3)
        with a1:
            if st.button("💾 Save override", width="stretch"):
                _update_profile_override(
                    row,
                    current_multiplier=new_multiplier,
                    recommended_action=new_action,
                )
                st.success("Profile row updated.")
                st.rerun()

        with a2:
            if st.button("♻️ Reset row", width="stretch"):
                _reset_profile_row(row)
                st.success("Profile row reset to defaults.")
                st.rerun()

        with a3:
            if st.button("🗑️ Delete row", width="stretch"):
                ok = _delete_profile_row(_profile_match_dict(row))
                if ok:
                    st.success("Profile row deleted.")
                else:
                    st.warning("Profile row not found.")
                st.rerun()

    st.markdown("### Bulk action")

    bulk_user = st.selectbox(
        "Delete all profile rows for student",
        usernames[1:] if len(usernames) > 1 else [],
        index=None,
        placeholder="Choose a student",
        key="nrp_bulk_user",
    )

    if bulk_user:
        if st.button("🗑️ Delete all rows for selected student", width="stretch"):
            n = _delete_profiles_for_user(bulk_user)
            st.success(f"Deleted {n} profile row(s) for {bulk_user}.")
            st.rerun()