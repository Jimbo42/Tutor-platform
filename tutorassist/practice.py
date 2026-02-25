import json
import requests
import streamlit as st
from streamlit import session_state as ss

from shared.content_renderer import render_questions_worksheet
from shared.google_db import get_sheet


TAB_INTERACTIVES = "Published_Interactives"


@st.cache_data(show_spinner="Loading Practice Library…")
def _load_published_interactives():
    ws = get_sheet(TAB_INTERACTIVES)
    rows = ws.get_all_records()
    return rows or []


def _drive_payload_from_row(r: dict) -> dict:
    file_id = str(r.get("drive_file_id") or "").strip()
    download_url = str(r.get("drive_download_url") or "").strip()

    # Fallback if your sheet doesn’t store download_url (recommended to store it!)
    # This usually works for public/shared-by-link files:
    if not download_url and file_id:
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    return {
        "provider": "gdrive",
        "file_id": file_id,
        "view_url": (r.get("drive_view_url") or "").strip(),
        "preview_url": (r.get("drive_preview_url") or "").strip(),
        "download_url": download_url,
        "public": str(r.get("public") or "").strip(),
        "title": (r.get("title") or "").strip(),
    }


@st.cache_data(show_spinner="Loading worksheet from Google Drive…")
def load_interactive_from_gdrive(download_url: str):
    try:
        r = requests.get(download_url, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}


def show_practice_library():
    # Default filter values (used before widgets render)
    f_subject = ss.get("pr_filter_subject", "All")
    f_grade = ss.get("pr_filter_grade", "All")
    f_type = ss.get("pr_filter_type", "All")

    st.markdown("## 📚 Practice Library")
    st.markdown(
        f"<div style='font-size:0.85rem;color:#888;'>"
        f"Filters → Subject: <b>{f_subject}</b> | "
        f"Grade: <b>{f_grade}</b> | "
        f"Type: <b>{f_type}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([5, 1], vertical_alignment="center")
    with c2:
        if st.button("🔄 Refresh", width="stretch", key="practice_refresh"):
            _load_published_interactives.clear()
            st.rerun()

    rows = _load_published_interactives()
    if not rows:
        st.info("No published items yet.")
        return

    # Convert to list of dicts for easier handling (normalize expected keys)
    records = []
    for r in rows:
        published_id = (r.get("published_id") or "").strip()
        if not published_id:
            continue

        def _s(x, default=""):
            return str(x if x is not None else default).strip()

        records.append({
            "published_id": published_id,
            "title": _s(r.get("title")) or "Untitled",
            "subject": _s(r.get("subject")) or "Other",
            "grade": _s(r.get("grade")) or "—",  # ✅ handles int grades
            "type": _s(r.get("interactive_type")) or "interactive",
            "created": _s(r.get("created_at")),
            "_row": r,
        })

    if not records:
        st.info("No usable interactive rows found (missing published_id).")
        return

    # ----------------------------
    # Filters
    # ----------------------------
    subjects = sorted(set(r["subject"] for r in records))
    grades = sorted(set(r["grade"] for r in records))
    types = sorted(set(r["type"] for r in records))

    with st.expander("🔎 Filter", expanded=False):
        st.caption("Use filters to narrow the library list.")
        c1, c2, c3 = st.columns(3)
        with c1:
            f_subject = st.selectbox("Subject", ["All"] + subjects, key="pr_filter_subject")
        with c2:
            f_grade = st.selectbox("Grade", ["All"] + grades, key="pr_filter_grade")
        with c3:
            f_type = st.selectbox("Type", ["All"] + types, key="pr_filter_type")

    def passes_filters(r):
        if f_subject != "All" and r["subject"] != f_subject:
            return False
        if f_grade != "All" and r["grade"] != f_grade:
            return False
        if f_type != "All" and r["type"] != f_type:
            return False
        return True

    filtered = [r for r in records if passes_filters(r)]
    if not filtered:
        st.warning("No items match these filters.")
        return

    st.divider()

    # ----------------------------
    # Direct link support (now uses published_id string)
    # ----------------------------
    qp = st.query_params
    forced_id = qp.get("item")  # string

    # ----------------------------
    # Build selector
    # ----------------------------
    label_map = {}
    labels = []

    for r in filtered:
        label = f"{r['title']}  —  {r['subject']} • Grade {r['grade']} • {r['type']}"
        labels.append(label)
        label_map[label] = r["published_id"]

    labels_with_placeholder = ["— Select an item —"] + labels

    default_index = 0
    if forced_id:
        forced_id = str(forced_id).strip()
        for i, r in enumerate(filtered):
            if r["published_id"] == forced_id:
                default_index = i + 1  # +1 placeholder
                break

    choice = st.selectbox(
        "Choose an item:",
        labels_with_placeholder,
        index=default_index
    )

    if choice == "— Select an item —":
        st.info("👆 Select a worksheet to begin.")
        return

    published_id = label_map[choice]
    # Only write query param if it changed (prevents double-rerun / double-click)
    if st.query_params.get("item") != str(published_id):
        st.query_params["item"] = str(published_id)

    # ----------------------------
    # Load item row
    # ----------------------------
    rec = next((r for r in records if r["published_id"] == published_id), None)
    if not rec:
        st.error("That item was not found in the spreadsheet.")
        return

    row = rec["_row"]

    st.divider()
    st.subheader(rec["title"])
    st.caption(f"{rec['subject']} • Grade {rec['grade']} • interactive")

    # ----------------------------
    # Render content (Drive-backed JSON)
    # ----------------------------
    payload = _drive_payload_from_row(row)
    download_url = payload.get("download_url")

    if not download_url:
        st.error("Interactive row is missing a Drive download_url (or drive_file_id).")
        st.json(row)
        return

    worksheet = load_interactive_from_gdrive(download_url)

    if isinstance(worksheet, dict) and worksheet.get("_error"):
        st.error(f"Could not load worksheet: {worksheet['_error']}")
        st.json(payload)
        return

    if isinstance(worksheet, dict) and worksheet.get("type") == "questions":
        render_questions_worksheet( worksheet, ws_key=published_id)
    else:
        st.warning("Downloaded JSON is not a questions worksheet.")
        st.json(worksheet)
