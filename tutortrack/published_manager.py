import io
import json
import streamlit as st
import pandas as pd
from streamlit import session_state as ss
from datetime import datetime

from shared.google_db import get_sheet, get_drive_service
from shared.content_renderer import render_questions_worksheet
from shared.google_db import (
    publish_item,
    upload_interactive_json,
    upload_pdf_bytes,
    download_drive_file_bytes,
    update_drive_file_bytes,
    safe_filename,
    delete_drive_file,
    update_cell_by_published_id,
)

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

REFERENCE_GENERATOR_IDS = [
    ("", "— No reference link —"),
    ("solving_eq_l1", "Solving Equations - Level 1"),
    ("solving_eq_l2", "Solving Equations - Level 2"),
    ("solving_eq_l3", "Solving Equations - Level 3"),
    ("solving_eq_l4", "Solving Equations - Level 4"),
    ("factoring_l1", "Factoring - Level 1"),
    ("factoring_l2", "Factoring - Level 2"),
    ("factoring_l3", "Factoring - Level 3"),
    ("factoring_l4", "Factoring - Level 4"),
    ("factoring_l5", "Factoring - Level 5"),
    ("factoring_l6", "Factoring - Level 6"),
    ("factoring_l7", "Factoring - Level 7"),
]

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

def _download_interactive_json(file_id: str) -> dict:
    raw = download_drive_file_bytes(file_id)
    return json.loads(raw.decode("utf-8"))
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
        ("generator_id", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    df["type"] = content_type
    return df

def _used_generator_ids_excluding(published_id: str | None = None) -> set[str]:
    rows = _load_tab_records(TAB_PDFS)
    used = set()

    for r in rows:
        pid = str(r.get("published_id", "")).strip()
        gid = str(r.get("generator_id", "")).strip()
        if not gid:
            continue
        if published_id and pid == published_id:
            continue
        used.add(gid)

    return used

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
            render_questions_worksheet(obj, ws_key=key_prefix)

        if item.get("drive_view_url"):
            st.link_button("🔗 Open in Drive", item["drive_view_url"], width="stretch")

    else:
        st.info("Unknown content type. Showing raw record:")
        st.json(item)

    st.divider()
    if st.button("Close", width="stretch"):
        ss.pm_preview_item = None
        ss.pm_preview_obj = None
        ss.pm_preview_error = None
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

            # Close preview + clear cache for this file before deleting
            if ss.get("pm_preview_item", {}).get("drive_file_id") == file_id:
                ss.pm_preview_item = None
                ss.pm_preview_obj = None
                ss.pm_preview_error = None

            if file_id:
                ss.pm_interactive_cache.pop(file_id, None)
                ss.pm_interactive_cache_meta.pop(file_id, None)

            # 1) Remove from catalog tab
            _delete_catalog_row(tab_name, published_id)

            # 2) Optionally delete Drive file
            if delete_drive_too and file_id:
                delete_drive_file(file_id)

            st.toast("Deleted", icon="🗑️")
            st.rerun()

        except Exception as e:
            st.error(f"Delete failed: {e}")

@st.dialog("✏️ Edit Interactive JSON", width="large")
def open_edit_interactive_json_dialog(item: dict):
    """
    Loads an interactive JSON file from Drive (by drive_file_id),
    lets you edit it, and saves back to the same file_id.
    """
    title = (item.get("title") or "").strip() or "Untitled"
    file_id = (item.get("drive_file_id") or "").strip()

    st.subheader(title)
    st.caption(f"Drive file_id: {file_id}")
    st.divider()

    if not file_id:
        st.error("This item is missing drive_file_id.")
        return

    # Load JSON bytes from Drive
    try:
        raw_bytes = download_drive_file_bytes(file_id)
        raw_text = raw_bytes.decode("utf-8", errors="replace")
        obj = json.loads(raw_text)
    except Exception as e:
        st.error(f"Could not load JSON from Drive: {e}")
        return

    edited = st.text_area(
        "Worksheet JSON",
        value=json.dumps(obj, ensure_ascii=False, indent=2),
        height=520,
    )

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("✅ Validate", width="stretch"):
            try:
                parsed = json.loads(edited)
                if not isinstance(parsed, dict):
                    st.error("Top-level JSON must be an object.")
                else:
                    st.success("Valid JSON ✅")
                    st.caption(
                        f"type={parsed.get('type')} • "
                        f"questions={len(parsed.get('questions', [])) if isinstance(parsed.get('questions'), list) else 'n/a'}"
                    )
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    with c2:
        if st.button("💾 Save to Drive", width="stretch"):
            try:
                parsed = json.loads(edited)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                return

            if not isinstance(parsed, dict):
                st.error("Top-level JSON must be an object.")
                return

            try:
                data = json.dumps(parsed, ensure_ascii=False, indent=2).encode("utf-8")
                update_drive_file_bytes(file_id=file_id, data=data, mime_type="application/json")
                st.toast("Saved to Drive ✅", icon="💾")
                st.rerun()
            except Exception as e:
                st.error(f"Save failed: {e}")

    with c3:
        if st.button("Close", width="stretch"):
            st.rerun()

@st.dialog("🔗 Assign Reference PDF", width="large")
def open_assign_generator_dialog(item: dict):
    published_id = str(item.get("published_id", "")).strip()
    title = str(item.get("title", "")).strip() or "Untitled"
    current_gid = str(item.get("generator_id", "")).strip()

    st.subheader(title)
    st.caption("Assign this PDF as the reference lesson for one practice generator.")
    st.divider()

    used_ids = _used_generator_ids_excluding(published_id)
    allowed = []
    for gid, label in REFERENCE_GENERATOR_IDS:
        if gid == "" or gid == current_gid or gid not in used_ids:
            allowed.append((gid, label))

    labels = [label for _, label in allowed]
    gid_by_label = {label: gid for gid, label in allowed}

    current_label = next((label for gid, label in allowed if gid == current_gid), labels[0])

    selected_label = st.selectbox(
        "Generator ID",
        labels,
        index=labels.index(current_label),
        key=f"pm_assign_gid_{published_id}",
    )

    selected_gid = gid_by_label[selected_label]

    st.caption("Only unassigned generator IDs are offered, so the relationship stays one-to-one.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("💾 Save assignment", width="stretch"):
            ok = update_cell_by_published_id(TAB_PDFS, published_id, "generator_id", selected_gid)
            if not ok:
                st.error("Could not update generator_id for this PDF.")
                return
            st.toast("Reference assignment saved ✅", icon="✅")
            st.rerun()

    with c2:
        if st.button("Clear assignment", width="stretch"):
            ok = update_cell_by_published_id(TAB_PDFS, published_id, "generator_id", "")
            if not ok:
                st.error("Could not clear generator_id for this PDF.")
                return
            st.toast("Reference assignment cleared", icon="🧹")
            st.rerun()

# -----------------------------
# Page
# -----------------------------
def show_published_manager():
    st.markdown("## 📦 Published Content Manager")

    with st.expander("➕ Import external JSON/PDF and publish", expanded=False):
        st.caption(
            "Upload a local .json (interactive) or .pdf (notes). It will upload to Drive and publish to the student catalog.")

        up = st.file_uploader("Choose file", type=["json", "pdf"], key="pm_import_file")

        c1, c2, c3 = st.columns(3)
        with c1:
            imp_title = st.text_input("Title", value="", key="pm_import_title")
        with c2:
            imp_subject = st.text_input("Subject", value="Other", key="pm_import_subject")
        with c3:
            imp_grade = st.text_input("Grade", value="—", key="pm_import_grade")

        make_public = st.checkbox("Make accessible to anyone with the link", value=True, key="pm_import_public")

        if up is None:
            st.info("Upload a .json or .pdf to enable publishing.")
        else:
            ext = up.name.lower().split(".")[-1]
            st.write(f"Detected: **.{ext}**")

            if st.button("📤 Upload + Publish", width="stretch", key="pm_import_publish"):
                if not imp_title.strip():
                    st.error("Title is required.")
                    return

                try:
                    if ext == "json":
                        # Validate JSON
                        raw = up.read()
                        obj = json.loads(raw.decode("utf-8"))

                        if not isinstance(obj, dict):
                            raise ValueError("Top-level JSON must be an object.")

                        if obj.get("type") != "questions":
                            st.warning("Normalizing worksheet JSON: forcing type='questions'")
                            obj["type"] = "questions"

                        payload = upload_interactive_json(
                            obj=obj,
                            title=imp_title.strip(),
                            filename=safe_filename(imp_title.strip(), ".json"),
                            make_public=make_public,
                        )

                        publish_item(
                            title=imp_title.strip(),
                            subject=imp_subject.strip(),
                            grade=imp_grade.strip(),
                            content_type="interactive",
                            content=payload,
                            source_app="TutorTrack",
                            notes=f"Imported upload: {up.name}",
                        )

                        st.toast("Imported + published interactive ✅", icon="✅")
                        st.rerun()

                    elif ext == "pdf":
                        pdf_bytes = up.read()

                        payload = upload_pdf_bytes(
                            pdf_bytes=pdf_bytes,
                            title=imp_title.strip(),
                            filename=safe_filename(imp_title.strip(), ".pdf"),
                            make_public=make_public,
                        )

                        publish_item(
                            title=imp_title.strip(),
                            subject=imp_subject.strip(),
                            grade=imp_grade.strip(),
                            content_type="pdf",
                            content=payload,
                            source_app="TutorTrack",
                            notes=f"Imported upload: {up.name}",
                        )

                        st.toast("Imported + published PDF ✅", icon="✅")
                        st.rerun()

                    else:
                        st.error("Unsupported file type.")
                except Exception as e:
                    st.error(f"Import/publish failed: {e}")

    # Load both tabs
    try:
        pdf_rows = _load_tab_records(TAB_PDFS)
    except Exception as e:
        st.error(f"Could not load '{TAB_PDFS}': {e}")
        pdf_rows = []

    try:
        int_rows = _load_tab_records(TAB_INTERACTIVES)
    except Exception as e:
        st.error(f"Could not load '{TAB_INTERACTIVES}': {e}")
        int_rows = []

    df_p = _normalize_catalog_records(pdf_rows, "pdf")
    df_i = _normalize_catalog_records(int_rows, "interactive")

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

            left, right = st.columns([4, 2], vertical_alignment="center")
            with left:
                st.markdown(label_html, unsafe_allow_html=True)
                # Quick links
                view_url = row.get("drive_view_url", "")
                if view_url:
                    st.link_button("Open in Drive", view_url, width="stretch")
            with right:
                # actions
                a1, a2, a3  = st.columns(3)
                with a1:
                    if st.button("👁", key=f"pm_prev_{i}", width="stretch", help="Preview"):
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
                    if st.button("🗑", key=f"pm_del_{i}", width="stretch", help="Delete"):
                        open_delete_dialog(row.to_dict())

                with a3:
                    if ctype == "interactive":
                        if st.button("✏️", key=f"pm_edit_{i}", width="stretch", help="Edit"):
                            open_edit_interactive_json_dialog(row.to_dict())
                    elif ctype == "pdf":
                        if st.button("🔗", key=f"pm_link_{i}", width="stretch", help="Assign reference PDF"):
                            open_assign_generator_dialog(row.to_dict())

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

