import io
import json
import streamlit as st
import pandas as pd
from streamlit import session_state as ss
from datetime import datetime

from shared.google_db import get_sheet, get_drive_service


# -----------------------------
# Sheet tabs (must exist)
# -----------------------------
TAB_PDFS = "Published_PDFs"
TAB_INTERACTIVES = "Published_Interactives"

ss.setdefault("pm_preview_item", None)
ss.setdefault("pm_preview_obj", None)
ss.setdefault("pm_preview_error", None)

# Cache for interactive JSON payloads (drive_file_id -> dict)
ss.setdefault("pm_interactive_cache", {})        # {file_id: obj}
ss.setdefault("pm_interactive_cache_meta", {})   # {file_id: {"loaded_at": "..."}}

# -----------------------------
# Drive helpers
# -----------------------------

def _load_interactive_json_cached(file_id: str, *, force_refresh: bool = False) -> dict:
    """
    Load interactive JSON from Drive with a session-state cache.

    - Cached by drive_file_id
    - force_refresh=True bypasses cache
    """
    cache = ss.get("pm_interactive_cache", {})
    meta = ss.get("pm_interactive_cache_meta", {})

    if not force_refresh and file_id in cache:
        return cache[file_id]

    obj = _download_interactive_json(file_id)
    cache[file_id] = obj
    meta[file_id] = {"loaded_at": datetime.now().isoformat(timespec="seconds")}

    ss.pm_interactive_cache = cache
    ss.pm_interactive_cache_meta = meta
    return obj

def _download_drive_file_bytes(file_id: str) -> bytes:
    """
    Download a Drive file as bytes using the Drive API client
    (which should be OAuth-based in your environment for My Drive uploads).
    """
    if not file_id:
        raise ValueError("Missing file_id")

    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)

    fh = io.BytesIO()
    from googleapiclient.http import MediaIoBaseDownload  # local import

    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue()


def _download_interactive_json(file_id: str) -> dict:
    raw = _download_drive_file_bytes(file_id)
    return json.loads(raw.decode("utf-8"))


def _delete_drive_file(file_id: str):
    """Best-effort delete. If permissions prevent it, we'll surface the error."""
    service = get_drive_service()
    service.files().delete(fileId=file_id).execute()


# -----------------------------
# Sheets helpers
# -----------------------------
def _load_tab_records(tab_name: str) -> list[dict]:
    """
    Return list of row dicts from a tab.
    Uses header row mapping. Blank sheet -> [].
    """
    ws = get_sheet(tab_name)
    records = ws.get_all_records()  # list[dict], keys from header row
    return records or []


def _normalize_catalog_records(records: list[dict], content_type: str) -> pd.DataFrame:
    """
    Normalize records into a consistent DataFrame.
    Expected header keys (recommended):
      published_id, title, subject, grade,
      drive_file_id, drive_view_url, drive_preview_url, drive_download_url,
      public, created_at, source_app, notes
    But we handle missing columns gracefully.
    """
    df = pd.DataFrame(records) if records else pd.DataFrame()

    # Standardize column names if someone used slightly different headers
    # (Add more aliases here if needed)
    rename_map = {
        "drive_fileid": "drive_file_id",
        "file_id": "drive_file_id",
        "view_url": "drive_view_url",
        "preview_url": "drive_preview_url",
        "download_url": "drive_download_url",
        "created": "created_at",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # Ensure required fields exist
    for col, default in [
        ("published_id", ""),
        ("title", ""),
        ("subject", "Other"),
        ("grade", "—"),
        ("drive_file_id", ""),
        ("drive_view_url", ""),
        ("drive_preview_url", ""),
        ("drive_download_url", ""),
        ("public", ""),
        ("created_at", ""),
        ("source_app", ""),
        ("notes", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    df["type"] = content_type
    return df


def _find_row_number_by_published_id(tab_name: str, published_id: str) -> int | None:
    """
    Find the 1-based row number in the sheet (including header row=1)
    for the row whose 'published_id' cell matches.
    Returns None if not found.
    """
    ws = get_sheet(tab_name)
    headers = ws.row_values(1)
    if not headers:
        return None

    try:
        col_idx = headers.index("published_id") + 1  # 1-based
    except ValueError:
        return None

    # Pull the whole column (excluding header) and search
    col_vals = ws.col_values(col_idx)  # includes header at index 0
    for i in range(2, len(col_vals) + 1):  # start at row 2
        if (col_vals[i - 1] or "").strip() == (published_id or "").strip():
            return i
    return None

def _delete_catalog_row(tab_name: str, published_id: str):
    ws = get_sheet(tab_name)
    row_num = _find_row_number_by_published_id(tab_name, published_id)
    if not row_num:
        raise ValueError(f"Could not find published_id='{published_id}' in tab '{tab_name}'.")
    ws.delete_rows(row_num)

def render_questions_interactive_preview(obj: dict, key_prefix: str = "iprev"):
    """
    Dialog-safe minimal interactive player for JSON like:
    {
      "type": "questions",
      "questions": [
        {"question": "...", "choices": [...], "correct_index": 0, "explanation": "..."},
        ...
      ]
    }

    IMPORTANT: No st.rerun() calls (dialogs can blank with forced reruns).
    Button clicks naturally rerun the app.
    """
    if not isinstance(obj, dict) or obj.get("type") != "questions":
        st.info("This interactive is not a 'questions' worksheet type. Showing JSON instead:")
        st.json(obj)
        return

    questions = obj.get("questions", [])
    if not isinstance(questions, list) or not questions:
        st.warning("No questions found in this interactive.")
        st.json(obj)
        return

    idx_key = f"{key_prefix}_idx"
    answered_key = f"{key_prefix}_answered"
    score_key = f"{key_prefix}_score"

    # Ensure keys exist
    ss.setdefault(idx_key, 0)
    ss.setdefault(answered_key, False)
    ss.setdefault(score_key, 0)

    # Clamp index
    i = int(ss[idx_key])
    i = max(0, min(i, len(questions) - 1))
    ss[idx_key] = i

    q = questions[i] if isinstance(questions[i], dict) else {}

    q_text = q.get("question") or q.get("prompt") or q.get("text") or f"Question {i+1}"
    choices = q.get("choices") or q.get("options") or []
    if not isinstance(choices, list):
        choices = []

    correct_index = q.get("correct_index", q.get("answer_index"))
    try:
        correct_index = int(correct_index) if correct_index is not None else None
    except Exception:
        correct_index = None

    explanation = q.get("explanation") or q.get("explain") or ""

    # Header
    c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="center")
    with c1:
        st.markdown(f"### Question {i+1} of {len(questions)}")
    with c2:
        st.metric("Score", int(ss[score_key]))
    with c3:
        if st.button("↩ Reset", width="stretch", key=f"{key_prefix}_reset"):
            ss[idx_key] = 0
            ss[answered_key] = False
            ss[score_key] = 0
            # just return; click triggers rerun
            return

    st.markdown(q_text)

    if not choices:
        st.warning("This question has no choices (cannot preview interactively).")
        st.json(q)
        return

    disabled = bool(ss[answered_key])

    # Use a per-question radio key so moving next/prev resets selection cleanly
    radio_key = f"{key_prefix}_choice_q{i}"
    sel = st.radio(
        "Choose one:",
        options=list(range(len(choices))),
        format_func=lambda k: f"{chr(65+k)}. {choices[k]}",
        key=radio_key,
        disabled=disabled,
    )

    # Actions
    a1, a2, a3 = st.columns([1, 1, 1])
    with a1:
        if st.button("✅ Submit", width="stretch", disabled=disabled, key=f"{key_prefix}_submit"):
            ss[answered_key] = True
            if correct_index is not None and int(sel) == correct_index:
                ss[score_key] = int(ss[score_key]) + 1
            return

    with a2:
        if st.button("⟵ Prev", width="stretch", disabled=(i == 0), key=f"{key_prefix}_prev"):
            ss[idx_key] = max(0, i - 1)
            ss[answered_key] = False
            return

    with a3:
        if st.button("Next ⟶", width="stretch", disabled=(i >= len(questions) - 1), key=f"{key_prefix}_next"):
            ss[idx_key] = min(len(questions) - 1, i + 1)
            ss[answered_key] = False
            return

    # Feedback
    if ss[answered_key]:
        if correct_index is None:
            st.info("No correct_index provided for this question.")
        else:
            if 0 <= correct_index < len(choices):
                correct_text = f"{chr(65+correct_index)}. {choices[correct_index]}"
            else:
                correct_text = "(correct_index out of range)"

            if int(sel) == correct_index:
                st.success(f"✅ Correct — {correct_text}")
            else:
                st.error(f"❌ Not quite. Correct answer: {correct_text}")

        if explanation:
            st.markdown("**Explanation**")
            st.write(explanation)

# -----------------------------
# Dialogs
# -----------------------------
@st.dialog("👁 Preview Published Item", width="large")
def open_preview_dialog():
    item = ss.get("pm_preview_item") or {}
    obj = ss.get("pm_preview_obj")
    err = ss.get("pm_preview_error")

    title = (item.get("title") or "").strip() or "Untitled"
    subject = (item.get("subject") or "").strip() or "Other"
    grade = (item.get("grade") or "").strip() or "—"
    ctype = (item.get("type") or "").strip().lower()

    st.subheader(title)
    st.caption(f"{subject} • Grade {grade} • {ctype.upper() if ctype else 'ITEM'}")
    st.divider()

    if err:
        st.error(err)

    if ctype == "pdf":
        view_url = item.get("drive_view_url") or ""
        preview_url = item.get("drive_preview_url") or ""

        c1, c2 = st.columns([1, 1])
        with c1:
            if view_url:
                st.link_button("🔗 Open PDF (Drive)", view_url, width="stretch")
        with c2:
            if item.get("drive_download_url"):
                st.link_button("⬇️ Download PDF", item["drive_download_url"], width="stretch")

        if preview_url:
            st.markdown(
                f"""
                <iframe src="{preview_url}" width="100%" height="650"
                        style="border:0;border-radius:10px;"></iframe>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No preview_url available for this PDF.")

    elif ctype == "interactive":
        file_id = (item.get("drive_file_id") or "").strip()

        # Controls row
        c1, c2 = st.columns([1, 1], vertical_alignment="center")

        with c1:
            if st.button("🔄 Refresh from Drive", width="stretch", key="pm_refresh_drive"):
                if not file_id:
                    ss.pm_preview_error = "Missing drive_file_id for interactive."
                else:
                    try:
                        with st.spinner("Refreshing interactive from Drive..."):
                            ss.pm_preview_obj = _load_interactive_json_cached(file_id, force_refresh=True)
                            ss.pm_preview_error = None
                    except Exception as e:
                        ss.pm_preview_error = f"Refresh failed: {e}"
                return  # dialog will rerender on the click

        with c2:
            if st.button("🧹 Clear cache", width="stretch", key="pm_clear_cache"):
                if file_id:
                    ss.pm_interactive_cache.pop(file_id, None)
                    ss.pm_interactive_cache_meta.pop(file_id, None)
                ss.pm_preview_obj = None
                st.toast("Cache cleared for this item")
                return

        # Show cache metadata (if present)
        meta = ss.get("pm_interactive_cache_meta", {}).get(file_id, {}) if file_id else {}
        loaded_at = meta.get("loaded_at")
        if loaded_at:
            st.caption(f"Cached copy loaded at: {loaded_at}")

        # ✅ Render from already-loaded object (no Drive download here)
        if obj is None:
            st.info("No interactive payload loaded.")
        else:
            key_prefix = f"iprev_{item.get('published_id', '')}"
            render_questions_interactive_preview(obj, key_prefix=key_prefix)

        if item.get("drive_view_url"):
            st.link_button("🔗 Open in Drive", item["drive_view_url"], width="stretch")

    else:
        st.info("Unknown content type. Showing raw record:")
        st.json(item)

    st.divider()
    if st.button("Close", width="stretch"):
        st.rerun()

@st.dialog("🗑 Delete Published Item", width="large")
def open_delete_dialog(item: dict):
    published_id = (item.get("published_id") or "").strip()
    title = (item.get("title") or "").strip() or "Untitled"
    ctype = (item.get("type") or "").strip().lower()
    file_id = (item.get("drive_file_id") or "").strip()

    tab_name = TAB_PDFS if ctype == "pdf" else TAB_INTERACTIVES if ctype == "interactive" else None
    if not tab_name:
        st.error("Unknown content type; cannot delete.")
        return

    st.warning(f"Delete **{title}** from the catalog?")
    delete_drive_too = st.checkbox(
        "Also delete the underlying Drive file (recommended to avoid orphan files)",
        value=True,
    )

    st.caption(f"Catalog tab: `{tab_name}`")
    if file_id:
        st.caption(f"Drive file_id: `{file_id}`")
    st.divider()

    if st.button("❌ Confirm Delete", width="stretch"):
        try:
            # 1) Remove from catalog tab
            _delete_catalog_row(tab_name, published_id)

            # 2) Optionally delete Drive file
            if delete_drive_too and file_id:
                _delete_drive_file(file_id)

            st.toast("Deleted", icon="🗑️")
            st.rerun()

        except Exception as e:
            st.error(f"Delete failed: {e}")


# -----------------------------
# Page
# -----------------------------
def show_published_manager():
    st.markdown("## 📦 Published Content Manager")

    # Load both tabs
    try:
        pdf_rows = _load_tab_records(TAB_PDFS)
    except Exception as e:
        st.error(f"Could not load '{TAB_PDFS}': {e}")
        pdf_rows = []

    try:
        инт_rows = _load_tab_records(TAB_INTERACTIVES)
    except Exception as e:
        st.error(f"Could not load '{TAB_INTERACTIVES}': {e}")
        инт_rows = []

    df_p = _normalize_catalog_records(pdf_rows, "pdf")
    df_i = _normalize_catalog_records(инт_rows, "interactive")

    df = pd.concat([df_p, df_i], ignore_index=True) if (not df_p.empty or not df_i.empty) else pd.DataFrame()

    if df.empty:
        st.info("No published items found in the spreadsheet tabs.")
        return

    # Normalize for filtering
    df["subject"] = df["subject"].fillna("Other").astype(str)
    df["grade"] = df["grade"].fillna("—").astype(str)
    df["type"] = df["type"].fillna("other").astype(str)

    # -----------------------------
    # Filters (collapsed by default)
    # -----------------------------
    all_subjects = sorted(df["subject"].unique().tolist())
    all_grades = sorted(df["grade"].unique().tolist(), key=lambda x: str(x))
    all_types = sorted(df["type"].unique().tolist())

    ss.setdefault("pm_search", "")
    ss.setdefault("pm_subject", "All")
    ss.setdefault("pm_grade", "All")
    ss.setdefault("pm_type", "All")

    with st.expander("🔎 Filters", expanded=False):
        c1, c2 = st.columns([2, 1], vertical_alignment="center")
        with c1:
            ss.pm_search = st.text_input(
                "Search (title / subject / grade / type)",
                value=ss.pm_search,
                placeholder="e.g., chemistry, grade 11, pdf, matter…",
                key="pm_search_input",
            )
        with c2:
            if st.button("Clear filters", width="stretch"):
                ss.pm_search = ""
                ss.pm_subject = "All"
                ss.pm_grade = "All"
                ss.pm_type = "All"
                st.rerun()

        c1, c2, c3 = st.columns(3)
        with c1:
            ss.pm_subject = st.selectbox(
                "Subject",
                ["All"] + all_subjects,
                index=(["All"] + all_subjects).index(ss.pm_subject) if ss.pm_subject in (["All"] + all_subjects) else 0,
                key="pm_subject_sel",
            )
        with c2:
            ss.pm_grade = st.selectbox(
                "Grade",
                ["All"] + all_grades,
                index=(["All"] + all_grades).index(ss.pm_grade) if ss.pm_grade in (["All"] + all_grades) else 0,
                key="pm_grade_sel",
            )
        with c3:
            ss.pm_type = st.selectbox(
                "Type",
                ["All"] + all_types,
                index=(["All"] + all_types).index(ss.pm_type) if ss.pm_type in (["All"] + all_types) else 0,
                key="pm_type_sel",
            )

    # Apply filters
    fdf = df.copy()

    if ss.pm_subject != "All":
        fdf = fdf[fdf["subject"] == ss.pm_subject]
    if ss.pm_grade != "All":
        fdf = fdf[fdf["grade"] == ss.pm_grade]
    if ss.pm_type != "All":
        fdf = fdf[fdf["type"] == ss.pm_type]

    q = (ss.pm_search or "").strip().lower()
    if q:
        def match_row(r):
            blob = f"{r.get('title','')} {r.get('subject','')} {r.get('grade','')} {r.get('type','')}".lower()
            return q in blob
        fdf = fdf[fdf.apply(match_row, axis=1)]

    # Sort newest first if created_at exists
    if "created_at" in fdf.columns:
        fdf["_created_sort"] = pd.to_datetime(fdf["created_at"], errors="coerce")
        fdf = fdf.sort_values("_created_sort", ascending=False).drop(columns=["_created_sort"], errors="ignore")

    st.caption(f"Showing **{len(fdf)}** of **{len(df)}** published items.")

    # -----------------------------
    # List
    # -----------------------------
    with st.expander("📚 Published Items", expanded=True):

        if fdf.empty:
            st.info("No items match the current filters.")
            return

        for i, row in fdf.reset_index(drop=True).iterrows():
            ctype = str(row.get("type", "")).lower()

            if ctype == "interactive":
                bg = "rgba(90, 160, 255, 0.18)"
                border = "rgba(90, 160, 255, 0.45)"
                tag = "INTERACTIVE"
            elif ctype == "pdf":
                bg = "rgba(80, 200, 120, 0.18)"
                border = "rgba(80, 200, 120, 0.45)"
                tag = "PDF"
            else:
                bg = "rgba(180, 180, 180, 0.18)"
                border = "rgba(180, 180, 180, 0.45)"
                tag = ctype.upper() if ctype else "OTHER"

            title = row.get("title", "Untitled")
            subject = row.get("subject", "Other")
            grade = row.get("grade", "—")
            created_at = row.get("created_at", "")

            label_html = f"""
            <div style="
                display:inline-block;
                padding:6px 10px;
                border-radius:10px;
                background:{bg};
                border:1px solid {border};
                line-height:1.2;">
              <span style="font-weight:600;">{title}</span>
              <span style="color:#666;"> — {subject} • Grade {grade} • {tag} • {created_at}</span>
            </div>
            """

            left, right = st.columns([6, 2], vertical_alignment="center")
            with left:
                st.markdown(label_html, unsafe_allow_html=True)
                # Quick links
                view_url = row.get("drive_view_url", "")
                if view_url:
                    st.markdown(f"[Open in Drive]({view_url})")
            with right:
                # actions
                a1, a2 = st.columns(2)
                with a1:
                    if st.button("👁 Preview", key=f"pm_prev_{i}", width="stretch"):
                        item = row.to_dict()
                        ss.pm_preview_item = item
                        ss.pm_preview_obj = None
                        ss.pm_preview_error = None

                        # Preload outside the dialog (prevents dialog blanking)
                        if str(item.get("type", "")).lower() == "interactive":
                            file_id = (item.get("drive_file_id") or "").strip()
                            if not file_id:
                                ss.pm_preview_error = "Missing drive_file_id for interactive."
                            else:
                                try:
                                    # Optional: allow refresh from UI later; for now just use cached
                                    with st.spinner("Loading interactive from Drive..."):
                                        ss.pm_preview_obj = _load_interactive_json_cached(file_id, force_refresh=False)
                                except Exception as e:
                                    ss.pm_preview_error = f"Failed to load interactive JSON from Drive: {e}"

                        open_preview_dialog()

                with a2:
                    if st.button("🗑 Delete", key=f"pm_del_{i}", width="stretch"):
                        open_delete_dialog(row.to_dict())

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
