import json
import streamlit as st

from shared.content_renderer import render_published_content
from shared.google_db import get_sheet


TAB_PDFS = "Published_PDFs"


@st.cache_data(show_spinner="Loading Notes Library…")
def _load_published_pdfs():
    ws = get_sheet(TAB_PDFS)
    rows = ws.get_all_records()  # list[dict] using header row
    return rows or []


def _drive_payload_from_row(r: dict) -> dict:
    # Build the same payload shape your renderer expects (provider=gdrive + urls)
    file_id = (r.get("drive_file_id") or "").strip()
    return {
        "provider": "gdrive",
        "file_id": file_id,
        "view_url": (r.get("drive_view_url") or "").strip(),
        "preview_url": (r.get("drive_preview_url") or "").strip(),
        "download_url": (r.get("drive_download_url") or "").strip(),
        "public": str(r.get("public") or "").strip(),
        "title": (r.get("title") or "").strip(),
    }


def show_notes_library():
    st.markdown("## 📘 Notes & Documents")
    st.caption("Choose a PDF to read while you work on practice questions.")

    c1, c2 = st.columns([5, 1], vertical_alignment="center")
    with c2:
        if st.button("🔄 Refresh", width="stretch", key="notes_refresh"):
            _load_published_pdfs.clear()
            st.rerun()

    rows = _load_published_pdfs()
    if not rows:
        st.info("No published notes yet.")
        return

    # Build labels + id map (published_id is string now)
    labels = []
    id_map = {}

    for r in rows:
        published_id = (r.get("published_id") or "").strip()
        title = (r.get("title") or "").strip() or "Untitled"
        subject = (r.get("subject") or "").strip() or "Other"
        grade = str(r.get("grade") or "").strip()

        if not published_id:
            continue

        label = f"{title} — {subject} • Grade {grade}"
        labels.append(label)
        id_map[label] = published_id

    if not labels:
        st.info("No usable PDF rows found (missing published_id).")
        return

    choice = st.selectbox(
        "Choose a document",
        ["— Select a document —"] + labels,
        index=0,
        key="notes_choice",
        label_visibility="collapsed",
    )

    if choice == "— Select a document —":
        st.info("👆 Select a document to view.")
        return

    sel_id = id_map[choice]
    row = next((r for r in rows if str(r.get("published_id", "")).strip() == sel_id), None)

    if not row:
        st.error("That document was not found in the spreadsheet.")
        return

    title = (row.get("title") or "").strip() or "Untitled"
    subject = (row.get("subject") or "").strip() or "Other"
    grade = str(r.get("grade") or "").strip()
    st.markdown(f"### {title}")
    st.caption(f"{subject} • Grade {grade} • pdf")

    payload = _drive_payload_from_row(row)

    # Basic sanity check
    if not payload.get("file_id") and not payload.get("preview_url") and not payload.get("view_url"):
        st.error("This row is missing Drive file info (drive_file_id / view/preview url).")
        st.json(row)
        return

    # Let shared renderer handle embedding / open button
    render_published_content(json.dumps(payload), content_type="pdf")
