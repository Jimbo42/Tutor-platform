"""
google_db.py
============

Single utility module for Google Drive + Google Sheets access in the Tutor project.

Goals
-----
- All Google auth + service creation lives here.
- All Drive file/folder IDs live in Streamlit secrets.toml (no hard-coded IDs in code).
- Provide small, composable helpers used by TutorTrack + TutorAssist.

Expected secrets.toml structure (recommended)
--------------------------------------------

# Service account JSON (same as you have now)
[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "..."
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."

# Google Sheets (existing in your project)
[gSheets]
spreadsheet_id = "YOUR_SPREADSHEET_ID"

# Google Drive folder IDs (RAW IDs, not URLs)
[gdrive]
interactives_folder_id = "RAW_FOLDER_ID_FOR_INTERACTIVES"
pdfs_folder_id = "RAW_FOLDER_ID_FOR_PDFS"

Optionally, you can also store:
[gdrive]
default_make_public = true

Notes
-----
- Uses Drive API v3 + gspread.
- Caches clients in st.session_state to avoid rebuilding.
"""

from __future__ import annotations

import io
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import streamlit as st

# Optional dependency (only needed for Sheets)
import gspread

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from google.oauth2.credentials import Credentials as UserCredentials
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError

# ---------------------------
# Published catalog (Google Sheets tabs)
# ---------------------------
PUBLISHED_PDFS_TAB = "Published_PDFs"
PUBLISHED_INTERACTIVES_TAB = "Published_Interactives"

class GoogleDriveUserFacingError(RuntimeError):
    """A clean, user-facing Drive error with actionable instructions."""
    pass

def _drive_error_message(prefix: str, detail: str, fix: str) -> str:
    return (
        f"{prefix}\n\n"
        f"Details: {detail}\n\n"
        f"Fix: {fix}\n"
    )


def _raise_drive_user_facing_error(exc: Exception, *, context: str):
    """
    Convert common google auth/drive exceptions into a friendly error.
    """
    # 1) OAuth token revoked/expired
    if isinstance(exc, RefreshError):
        msg = _drive_error_message(
            prefix=f"Google Drive authorization failed while {context}.",
            detail="Your OAuth token was expired or revoked (invalid_grant).",
            fix=(
                "Generate a NEW refresh_token and update secrets.toml [gdrive_oauth].\n"
                "Then restart Streamlit and try again."
            ),
        )
        raise GoogleDriveUserFacingError(msg) from exc

    # 2) Drive API HttpError (permissions, quota, etc.)
    if isinstance(exc, HttpError):
        # Try to extract readable content
        try:
            raw = exc.content.decode("utf-8", errors="ignore") if hasattr(exc, "content") else str(exc)
        except Exception:
            raw = str(exc)

        status = getattr(exc.resp, "status", None)

        # Service account quota issue
        if status == 403 and ("Service Accounts do not have storage quota" in raw or "storageQuotaExceeded" in raw):
            msg = _drive_error_message(
                prefix=f"Google Drive upload blocked while {context}.",
                detail="Service accounts cannot upload into My Drive because they have no storage quota.",
                fix=(
                    "Use personal OAuth for Drive uploads (recommended for My Drive), OR\n"
                    "upload into a Shared Drive (Workspace), OR\n"
                    "enable domain-wide delegation (Workspace admin)."
                ),
            )
            raise GoogleDriveUserFacingError(msg) from exc

        # Public sharing disallowed / permissions errors
        if status == 403 and ("insufficientFilePermissions" in raw or "cannotShare" in raw or "forbidden" in raw):
            msg = _drive_error_message(
                prefix=f"Google Drive permission error while {context}.",
                detail=raw[:4000],
                fix=(
                    "Confirm you have access to the target folder and that link sharing is allowed.\n"
                    "If using service account, confirm the folder is shared with the service account email."
                ),
            )
            raise GoogleDriveUserFacingError(msg) from exc

        # Not found (wrong folder id / not shared)
        if status == 404:
            msg = _drive_error_message(
                prefix=f"Google Drive folder/file not found while {context}.",
                detail=raw[:2000],
                fix="Verify the folder_id / file_id is correct and shared with the account being used.",
            )
            raise GoogleDriveUserFacingError(msg) from exc

        # Fallback for other HttpErrors
        msg = _drive_error_message(
            prefix=f"Google Drive error while {context}.",
            detail=raw[:4000],
            fix="See details above. If this persists, re-auth (OAuth) or verify folder permissions.",
        )
        raise GoogleDriveUserFacingError(msg) from exc

    # Unknown
    raise GoogleDriveUserFacingError(
        _drive_error_message(
            prefix=f"Unexpected Google Drive error while {context}.",
            detail=str(exc),
            fix="Check your Google configuration and credentials.",
        )
    ) from exc

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _make_published_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

def append_row_by_header(tab_name: str, row_dict: dict):
    """
    Append a row to a sheet tab using the header row (row 1) to determine column order.

    Any keys in row_dict not present in the header are ignored.
    Any header columns not present in row_dict are written as blank.
    """
    ws = get_sheet(tab_name)

    header_cache_key = f"gsheet_headers::{tab_name}"
    headers = st.session_state.get(header_cache_key)

    if not headers:
        headers = ws.row_values(1)
        if not headers:
            raise ValueError(f"Sheet '{tab_name}' has no header row (row 1).")
        st.session_state[header_cache_key] = headers

    row = [row_dict.get(h, "") for h in headers]
    ws.append_row(row, value_input_option="USER_ENTERED")

def _header_index_map(ws):
    headers = ws.row_values(1)
    if not headers:
        raise ValueError(f"Sheet '{ws.title}' has no header row (row 1).")
    return {h: i + 1 for i, h in enumerate(headers)}, headers


def row_exists_by_header(tab_name: str, header_name: str, value: str) -> bool:
    """
    Return True if any data row in tab_name already contains `value`
    under column `header_name`.
    """
    if value is None or str(value).strip() == "":
        return False

    ws = get_sheet(tab_name)
    idx_map, _ = _header_index_map(ws)

    if header_name not in idx_map:
        raise ValueError(f"Sheet '{tab_name}' is missing required header '{header_name}'.")

    col_values = ws.col_values(idx_map[header_name])
    target = str(value).strip()

    # skip header row
    for cell in col_values[1:]:
        if str(cell).strip() == target:
            return True
    return False


def append_row_by_header_unique(tab_name: str, row_dict: dict, unique_header: str):
    """
    Append row only if unique_header value is not already present.
    Returns True if appended, False if skipped as duplicate.
    """
    unique_value = str(row_dict.get(unique_header, "")).strip()
    if not unique_value:
        raise ValueError(f"row_dict must include non-empty '{unique_header}'")

    if row_exists_by_header(tab_name, unique_header, unique_value):
        return False

    append_row_by_header(tab_name, row_dict)
    return True

def publish_item(
    *,
    title: str,
    subject: str,
    grade: str,
    content_type: str,  # "pdf" | "interactive"
    content: dict,
    source_app: str | None = None,
    notes: str = "",
) -> dict:
    """
    Write a published catalog entry to TutorAssist_Data spreadsheet.

    - PDFs go to Published_PDFs
    - Interactives go to Published_Interactives

    `content` should be the payload returned by upload_pdf_bytes / upload_interactive_json.
    """
    ct = (content_type or "").strip().lower()
    if ct not in ("pdf", "interactive"):
        raise ValueError("content_type must be 'pdf' or 'interactive'")

    tab = PUBLISHED_PDFS_TAB if ct == "pdf" else PUBLISHED_INTERACTIVES_TAB
    published_id = _make_published_id(ct)

    # Pull common fields from the Drive payload
    file_id = (content or {}).get("file_id", "")
    view_url = (content or {}).get("view_url", "")
    download_url = (content or {}).get("download_url", "")
    preview_url = (content or {}).get("preview_url", "")
    public_flag = True if (getattr(_get_google_config(), "default_make_public", False)) else False

    # If you pass make_public=False, you can override by setting content["public"]=False, etc.
    # (optional)
    if isinstance(content, dict) and "public" in content:
        public_flag = bool(content["public"])

    row = {
        "published_id": published_id,
        "title": (title or "").strip(),
        "subject": (subject or "").strip(),
        "grade": (grade or "").strip(),
        "drive_file_id": file_id,
        "drive_view_url": view_url,
        "drive_download_url": download_url,
        "drive_preview_url": preview_url,
        "public": str(bool(public_flag)).upper(),  # TRUE/FALSE
        "created_at": _now_iso(),
        "source_app": source_app or "TutorTrack",
        "notes": notes or "",
        # Optional extras if your headers include them:
        "filename": (content or {}).get("filename", ""),
        "folder": (content or {}).get("folder", ""),
        "provider": (content or {}).get("provider", ""),
    }

    # Interactive extras if present
    if ct == "interactive" and isinstance(content, dict):
        # If your JSON object includes metadata, you can add it here later.
        row.setdefault("interactive_type", "")
        row.setdefault("questions_count", "")

    append_row_by_header(tab, row)
    return row

# ---------------------------
# Regex helpers (Drive links)
# ---------------------------
_DRIVE_FILE_ID = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")
_DRIVE_UC_ID = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")
_DRIVE_OPEN_ID = re.compile(r"/open\?id=([a-zA-Z0-9_-]+)")


def extract_gdrive_file_id(url_or_id: str) -> str | None:
    """Extract a Google Drive file_id from common Drive URLs, or accept a raw id."""
    if not url_or_id:
        return None
    s = url_or_id.strip()

    m = _DRIVE_FILE_ID.search(s)
    if m:
        return m.group(1)

    m = _DRIVE_UC_ID.search(s)
    if m:
        return m.group(1)

    m = _DRIVE_OPEN_ID.search(s)
    if m:
        return m.group(1)

    # allow raw id (Drive IDs are typically 20+ chars of URL-safe base64-ish)
    if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", s):
        return s

    return None


def gdrive_urls(file_id: str) -> dict[str, str]:
    """Standard urls for a Drive file_id."""
    return {
        "view_url": f"https://drive.google.com/file/d/{file_id}/view",
        "preview_url": f"https://drive.google.com/file/d/{file_id}/preview",
        "download_url": f"https://drive.google.com/uc?export=download&id={file_id}",
    }


def safe_filename(title: str, ext: str, max_len: int = 80) -> str:
    """Conservative filename builder (Drive-friendly)."""
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", (title or "").strip())[:max_len].strip("_")
    if not base:
        base = "untitled"
    if not ext.startswith("."):
        ext = "." + ext
    return f"{base}{ext}"


# ---------------------------
# Secrets / config
# ---------------------------
@dataclass(frozen=True)
class GoogleConfig:
    spreadsheet_id: str | None
    interactives_folder_id: str | None
    pdfs_folder_id: str | None
    default_make_public: bool = True


def _get_google_config() -> GoogleConfig:
    # Backward-compatible with your existing structure:
    # - spreadsheet_id under st.secrets["gSheets"]["spreadsheet_id"]
    # - gdrive folder ids under st.secrets["gdrive"][...]
    gSheets = st.secrets.get("gSheets", {})
    gdrive = st.secrets.get("gdrive", {})

    spreadsheet_id = gSheets.get("spreadsheet_id") or gSheets.get("spreadsheetId")

    interactives_folder_id = (
        gdrive.get("interactives_folder_id")
        or gdrive.get("INTERACTIVES_FOLDER_ID")
        or gdrive.get("interactivesFolderId")
    )
    pdfs_folder_id = (
        gdrive.get("pdfs_folder_id")
        or gdrive.get("PDFS_FOLDER_ID")
        or gdrive.get("pdfsFolderId")
    )

    default_make_public = bool(gdrive.get("default_make_public", True))

    return GoogleConfig(
        spreadsheet_id=spreadsheet_id,
        interactives_folder_id=interactives_folder_id,
        pdfs_folder_id=pdfs_folder_id,
        default_make_public=default_make_public,
    )


# ---------------------------
# Auth / clients
# ---------------------------
def get_credentials(
    *,
    scopes: Optional[list[str]] = None,
) -> Credentials:
    """
    Build service-account credentials from Streamlit secrets.

    Default scopes cover both Sheets + Drive.
    """
    if scopes is None:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]

    if "gcp_service_account" not in st.secrets:
        raise KeyError("Missing st.secrets['gcp_service_account'] (service account JSON)")

    return Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes,
    )


def get_drive_service():
    """
    Cached Drive service client (Drive API v3).

    - Uses USER OAuth (refresh_token) for Drive uploads to My Drive (uses your quota).
    - Falls back to service account only if OAuth isn't configured.
    - Provides a clear error when refresh token is revoked/expired.
    """
    if "drive_service" in st.session_state:
        return st.session_state["drive_service"]

    # ✅ Prefer OAuth user creds for My Drive uploads
    if "gdrive_oauth" in st.secrets:
        cfg = st.secrets["gdrive_oauth"]
        try:
            creds = UserCredentials(
                token=None,
                refresh_token=cfg.get("refresh_token"),
                token_uri=cfg.get("token_uri", "https://oauth2.googleapis.com/token"),
                client_id=cfg["client_id"],
                client_secret=cfg["client_secret"],
                # Use full drive scope because you also set permissions
                scopes=["https://www.googleapis.com/auth/drive"],
            )

            if not creds.refresh_token:
                raise RefreshError("Missing refresh_token in [gdrive_oauth].")

            creds.refresh(Request())

            st.session_state["drive_service"] = build(
                "drive", "v3", credentials=creds, cache_discovery=False
            )
            return st.session_state["drive_service"]

        except Exception as e:
            _raise_drive_user_facing_error(e, context="initializing Drive service (OAuth refresh)")

    # Fallback: service account (NOTE: uploads to My Drive may fail with quota error)
    try:
        st.session_state["drive_service"] = build(
            "drive",
            "v3",
            credentials=get_credentials(scopes=["https://www.googleapis.com/auth/drive"]),
            cache_discovery=False,
        )
        return st.session_state["drive_service"]
    except Exception as e:
        _raise_drive_user_facing_error(e, context="initializing Drive service (service account)")

def get_gspread_client():
    """Cached gspread client for Google Sheets."""
    if "gspread_client" not in st.session_state:
        creds = get_credentials(scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        st.session_state["gspread_client"] = gspread.authorize(creds)
    return st.session_state["gspread_client"]


# ---------------------------
# Google Sheets helpers
# ---------------------------
def get_sheet(sheet_name: str):
    """
    Return a cached gspread Worksheet by name from the configured spreadsheet.
    """
    cache_key = f"gsheet_ws::{sheet_name}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    cfg = _get_google_config()
    if not cfg.spreadsheet_id:
        raise ValueError("Missing gSheets.spreadsheet_id in secrets.toml")

    client = get_gspread_client()
    sh = client.open_by_key(cfg.spreadsheet_id)
    ws = sh.worksheet(sheet_name)

    st.session_state[cache_key] = ws
    return ws

def read_sheet_as_df(sheet_name: str) -> pd.DataFrame:
    """
    Read a Google Sheet tab into a pandas DataFrame using row 1 as headers.
    Blank sheet -> empty DataFrame.
    """
    ws = get_sheet(sheet_name)
    values = ws.get_all_values()

    if not values:
        return pd.DataFrame()

    headers = values[0]
    rows = values[1:] if len(values) > 1 else []

    # pad short rows to header length
    normalized_rows = []
    for row in rows:
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))
        elif len(row) > len(headers):
            row = row[:len(headers)]
        normalized_rows.append(row)

    return pd.DataFrame(normalized_rows, columns=headers)

def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in ("true", "1", "yes", "y")

def _to_int(v, default: int = 0) -> int:
    try:
        if v in ("", None):
            return default
        return int(float(v))
    except Exception:
        return default

def _to_float(v, default: float = 0.0) -> float:
    try:
        if v in ("", None):
            return default
        return float(v)
    except Exception:
        return default

def _clean_str(v) -> str:
    return "" if v is None else str(v).strip()

def _worksheet_records(sheet_name: str) -> list[dict]:
    """
    Return all data rows as list[dict] using row 1 as headers.
    Blank sheet -> [].
    """
    ws = get_sheet(sheet_name)
    try:
        return ws.get_all_records()
    except Exception:
        df = read_sheet_as_df(sheet_name)
        return [] if df.empty else df.fillna("").to_dict("records")

def find_row_number_by_values(tab_name: str, match_dict: dict) -> int | None:
    """
    Find the first worksheet row number (2-based) whose cells match all supplied header/value pairs.
    Returns None if not found.
    """
    ws = get_sheet(tab_name)
    idx_map, headers = _header_index_map(ws)

    required_headers = list(match_dict.keys())
    for h in required_headers:
        if h not in idx_map:
            raise ValueError(f"Sheet '{tab_name}' is missing required header '{h}'.")

    all_values = ws.get_all_values()
    if len(all_values) <= 1:
        return None

    for row_num, row in enumerate(all_values[1:], start=2):
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))
        row_map = {headers[i]: row[i] for i in range(len(headers))}

        ok = True
        for h, expected in match_dict.items():
            if _clean_str(row_map.get(h, "")) != _clean_str(expected):
                ok = False
                break

        if ok:
            return row_num

    return None

def upsert_row_by_headers(tab_name: str, match_dict: dict, row_dict: dict):
    """
    Update an existing row matched by match_dict, or append a new row if not found.
    Writes values using the sheet header order.
    """
    ws = get_sheet(tab_name)
    idx_map, headers = _header_index_map(ws)

    row_values = [row_dict.get(h, "") for h in headers]
    row_num = find_row_number_by_values(tab_name, match_dict)

    if row_num is None:
        ws.append_row(row_values, value_input_option="USER_ENTERED")
        return "inserted"

    start_col = 1
    end_col = len(headers)
    cell_range = gspread.utils.rowcol_to_a1(row_num, start_col) + ":" + gspread.utils.rowcol_to_a1(row_num, end_col)
    ws.update(cell_range, [row_values], value_input_option="USER_ENTERED")
    return "updated"

def update_cell_by_published_id(tab_name: str, published_id: str, header_name: str, value: str):
    """
    Update one cell on a sheet row identified by published_id.
    """
    ws = get_sheet(tab_name)
    idx_map, _ = _header_index_map(ws)

    if "published_id" not in idx_map:
        raise ValueError(f"Sheet '{tab_name}' is missing required header 'published_id'.")
    if header_name not in idx_map:
        raise ValueError(f"Sheet '{tab_name}' is missing required header '{header_name}'.")

    published_col = idx_map["published_id"]
    target_col = idx_map[header_name]

    col_vals = ws.col_values(published_col)
    target = str(published_id or "").strip()

    for row_num in range(2, len(col_vals) + 1):
        if str(col_vals[row_num - 1]).strip() == target:
            ws.update_cell(row_num, target_col, value or "")
            return True

    return False


def get_published_pdf_by_generator_id(generator_id: str) -> dict | None:
    """
    Return the first Published_PDFs row matching generator_id, else None.
    """
    if not generator_id or not str(generator_id).strip():
        return None

    df = read_sheet_as_df(PUBLISHED_PDFS_TAB)
    if df.empty or "generator_id" not in df.columns:
        return None

    g = str(generator_id).strip()
    matches = df[df["generator_id"].astype(str).str.strip() == g]

    if matches.empty:
        return None

    row = matches.iloc[0].to_dict()
    return row


def get_published_pdf_preview_url_by_generator_id(generator_id: str) -> str | None:
    row = get_published_pdf_by_generator_id(generator_id)
    if not row:
        return None

    preview_url = str(row.get("drive_preview_url", "")).strip()
    if preview_url:
        return preview_url

    file_id = str(row.get("drive_file_id", "")).strip()
    if file_id:
        return gdrive_urls(file_id)["preview_url"]

    return None

def append_row(sheet_name: str, values: list[Any]):
    ws = get_sheet(sheet_name)
    ws.append_row(values)

def append_rows(sheet_name: str, rows: list[list[Any]], value_input_option: str = "USER_ENTERED"):
    """
    Append multiple rows in one API call.
    """
    if not rows:
        return

    ws = get_sheet(sheet_name)
    ws.append_rows(rows, value_input_option=value_input_option)

def append_score(username: str, skill: str, quiz_id: str, score: int, max_score: int, details: str | None = None):
    append_row(
        "scores",
        [
            datetime.now().isoformat(),
            username,
            skill,
            quiz_id,
            int(score),
            int(max_score),
            details or "",
        ],
    )

def build_numerace_attempt_row(
    *,
    username,
    round_key,
    round_id,
    attempt_id,
    question_seq,
    question_id,
    question_title,
    domain,
    skill,
    subskill,
    difficulty,
    mastery_group,
    correct,
    missed,
    attempts_on_question,
    response_time,
    selected_answer,
    correct_answer,
    choice_count,
    prompt_text,
    generated_values_json,
    tags_csv,
):
    return [
        datetime.now().isoformat(),
        username,
        round_key,
        round_id,
        attempt_id,
        question_seq,
        question_id,
        question_title,
        domain,
        skill,
        subskill,
        difficulty,
        mastery_group,
        correct,
        missed,
        attempts_on_question,
        response_time,
        selected_answer,
        correct_answer,
        choice_count,
        prompt_text,
        generated_values_json,
        tags_csv,
    ]

def append_numerace_attempt_rows(rows: list[dict]):
    payload = []
    for row in rows:
        payload.append(
            build_numerace_attempt_row(
                username=row.get("username", "unknown"),
                round_key=row.get("round_key", ""),
                round_id=row.get("round_id", ""),
                attempt_id=row.get("attempt_id", ""),
                question_seq=int(row.get("question_seq", 0)),
                question_id=row.get("question_id", ""),
                question_title=row.get("question_title", ""),
                domain=row.get("domain", ""),
                skill=row.get("skill", ""),
                subskill=row.get("subskill", ""),
                difficulty=row.get("difficulty", ""),
                mastery_group=row.get("mastery_group", ""),
                correct=bool(row.get("correct", False)),
                missed=bool(row.get("missed", False)),
                attempts_on_question=int(row.get("attempts_on_question", 0)),
                response_time=float(row.get("response_time", 0.0)),
                selected_answer=str(row.get("selected_answer", "")),
                correct_answer=str(row.get("correct_answer", "")),
                choice_count=int(row.get("choice_count", 0)),
                prompt_text=row.get("prompt_text", ""),
                generated_values_json=row.get("generated_values_json", "{}"),
                tags_csv=row.get("tags_csv", ""),
            )
        )

    append_rows("numerace_attempts", payload)

def append_numerace_round(
    *,
    username: str,
    round_key: str,
    round_id: str,
    game_name: str,
    questions_served: int,
    correct: int,
    incorrect: int,
    missed: int,
    attempts_total: int,
    round_time: float,
    average_response_time: float,
    accuracy: float,
    score: int,
    completed: bool,
    start_difficulty_mix: str = "",
    notes: str = "",
):
    row = {
        "timestamp": datetime.now().isoformat(),
        "username": username,
        "round_key": round_key,
        "round_id": round_id,
        "game_name": game_name,
        "questions_served": int(questions_served),
        "correct": int(correct),
        "incorrect": int(incorrect),
        "missed": int(missed),
        "attempts_total": int(attempts_total),
        "round_time": float(round_time),
        "average_response_time": float(average_response_time),
        "accuracy": float(accuracy),
        "score": int(score),
        "completed": bool(completed),
        "start_difficulty_mix": start_difficulty_mix,
        "notes": notes,
    }

    return append_row_by_header_unique("numerace_rounds", row, "round_key")

def append_numerace_attempt(
    *,
    attempt_id,
    username,
    round_key,
    round_id,
    question_seq,
    question_id,
    question_title,
    domain,
    skill,
    subskill,
    difficulty,
    mastery_group,
    correct,
    missed,
    attempts_on_question,
    response_time,
    selected_answer,
    correct_answer,
    choice_count,
    prompt_text,
    generated_values_json,
    tags_csv,
):
    row = {
        "timestamp": datetime.now().isoformat(),
        "attempt_id": attempt_id,
        "username": username,
        "round_key": round_key,
        "round_id": round_id,
        "question_seq": question_seq,
        "question_id": question_id,
        "question_title": question_title,
        "domain": domain,
        "skill": skill,
        "subskill": subskill,
        "difficulty": difficulty,
        "mastery_group": mastery_group,
        "correct": correct,
        "missed": missed,
        "attempts_on_question": attempts_on_question,
        "response_time": response_time,
        "selected_answer": selected_answer,
        "correct_answer": correct_answer,
        "choice_count": choice_count,
        "prompt_text": prompt_text,
        "generated_values_json": generated_values_json,
        "tags_csv": tags_csv,
    }

    return append_row_by_header_unique("numerace_attempts", row, "attempt_id")

def get_numerace_attempt_rows(username: str) -> list[dict]:
    """
    Read all numerace_attempts rows for one username.
    Returns normalized Python types for the main numeric/bool fields.
    """
    username = _clean_str(username)
    if not username:
        return []

    rows = _worksheet_records("numerace_attempts")
    out = []

    for r in rows:
        if _clean_str(r.get("username", "")) != username:
            continue

        out.append({
            "timestamp": _clean_str(r.get("timestamp", "")),
            "username": _clean_str(r.get("username", "")),
            "round_key": _clean_str(r.get("round_key", "")),
            "round_id": _clean_str(r.get("round_id", "")),
            "attempt_id": _clean_str(r.get("attempt_id", "")),
            "question_seq": _to_int(r.get("question_seq", 0)),
            "question_id": _clean_str(r.get("question_id", "")),
            "question_title": _clean_str(r.get("question_title", "")),
            "domain": _clean_str(r.get("domain", "")),
            "skill": _clean_str(r.get("skill", "")),
            "subskill": _clean_str(r.get("subskill", "")),
            "difficulty": _clean_str(r.get("difficulty", "")),
            "mastery_group": _clean_str(r.get("mastery_group", "")),
            "correct": _to_bool(r.get("correct", False)),
            "missed": _to_bool(r.get("missed", False)),
            "attempts_on_question": _to_int(r.get("attempts_on_question", 0)),
            "response_time": _to_float(r.get("response_time", 0.0)),
            "selected_answer": _clean_str(r.get("selected_answer", "")),
            "correct_answer": _clean_str(r.get("correct_answer", "")),
            "choice_count": _to_int(r.get("choice_count", 0)),
            "prompt_text": _clean_str(r.get("prompt_text", "")),
            "generated_values_json": _clean_str(r.get("generated_values_json", "{}")),
            "tags_csv": _clean_str(r.get("tags_csv", "")),
        })

    return out

def get_numerace_user_profile_rows_with_row_numbers(username: str) -> list[dict]:
    """
    Read all numerace_user_profile rows for one username, including worksheet row numbers.
    This performs one sheet read and returns normalized rows plus _row_number.
    """
    username = _clean_str(username)
    if not username:
        return []

    ws = get_sheet("numerace_user_profile")
    all_values = ws.get_all_values()
    if len(all_values) <= 1:
        return []

    headers = all_values[0]
    out = []

    for row_num, row in enumerate(all_values[1:], start=2):
        if len(row) < len(headers):
            row = row + [""] * (len(headers) - len(row))
        elif len(row) > len(headers):
            row = row[:len(headers)]

        r = {headers[i]: row[i] for i in range(len(headers))}
        if _clean_str(r.get("username", "")) != username:
            continue

        out.append({
            "_row_number": row_num,
            "timestamp_updated": _clean_str(r.get("timestamp_updated", "")),
            "username": _clean_str(r.get("username", "")),
            "domain": _clean_str(r.get("domain", "")),
            "skill": _clean_str(r.get("skill", "")),
            "subskill": _clean_str(r.get("subskill", "")),
            "mastery_group": _clean_str(r.get("mastery_group", "")),
            "questions_seen": _to_int(r.get("questions_seen", 0)),
            "correct_count": _to_int(r.get("correct_count", 0)),
            "missed_count": _to_int(r.get("missed_count", 0)),
            "incorrect_count": _to_int(r.get("incorrect_count", 0)),
            "accuracy": _to_float(r.get("accuracy", 0.0)),
            "avg_response_time": _to_float(r.get("avg_response_time", 0.0)),
            "recent_accuracy": _to_float(r.get("recent_accuracy", 0.0)),
            "recent_avg_response_time": _to_float(r.get("recent_avg_response_time", 0.0)),
            "current_multiplier": _to_float(r.get("current_multiplier", 1.0)),
            "recommended_action": _clean_str(r.get("recommended_action", "")),
            "last_seen": _clean_str(r.get("last_seen", "")),
        })

    return out

# def get_numerace_user_profile_rows(username: str) -> list[dict]:
#     """
#     Read all numerace_user_profile rows for one username.
#     """
#     username = _clean_str(username)
#     if not username:
#         return []
#
#     rows = _worksheet_records("numerace_user_profile")
#     out = []
#
#     for r in rows:
#         if _clean_str(r.get("username", "")) != username:
#             continue
#
#         out.append({
#             "timestamp_updated": _clean_str(r.get("timestamp_updated", "")),
#             "username": _clean_str(r.get("username", "")),
#             "domain": _clean_str(r.get("domain", "")),
#             "skill": _clean_str(r.get("skill", "")),
#             "subskill": _clean_str(r.get("subskill", "")),
#             "mastery_group": _clean_str(r.get("mastery_group", "")),
#             "questions_seen": _to_int(r.get("questions_seen", 0)),
#             "correct_count": _to_int(r.get("correct_count", 0)),
#             "missed_count": _to_int(r.get("missed_count", 0)),
#             "incorrect_count": _to_int(r.get("incorrect_count", 0)),
#             "accuracy": _to_float(r.get("accuracy", 0.0)),
#             "avg_response_time": _to_float(r.get("avg_response_time", 0.0)),
#             "recent_accuracy": _to_float(r.get("recent_accuracy", 0.0)),
#             "recent_avg_response_time": _to_float(r.get("recent_avg_response_time", 0.0)),
#             "current_multiplier": _to_float(r.get("current_multiplier", 1.0)),
#             "recommended_action": _clean_str(r.get("recommended_action", "")),
#             "last_seen": _clean_str(r.get("last_seen", "")),
#         })
#
#     return out

def upsert_numerace_user_profile_rows_fast(rows: list[dict], existing_row_map: dict | None = None) -> dict:
    """
    Upsert profile rows using an in-memory existing_row_map:
    key = (username, domain, skill, subskill, mastery_group) -> worksheet row number

    Returns updated row map.
    """
    if not rows:
        return existing_row_map or {}

    ws = get_sheet("numerace_user_profile")
    idx_map, headers = _header_index_map(ws)
    row_map = dict(existing_row_map or {})

    ts_now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    append_payload = []
    append_keys = []

    for row in rows:
        username = _clean_str(row.get("username", ""))
        domain = _clean_str(row.get("domain", ""))
        skill = _clean_str(row.get("skill", ""))
        subskill = _clean_str(row.get("subskill", ""))
        mastery_group = _clean_str(row.get("mastery_group", ""))

        if not username:
            continue

        key = (username, domain, skill, subskill, mastery_group)

        row_to_write = {
            "timestamp_updated": _clean_str(row.get("timestamp_updated", "")) or ts_now,
            "username": username,
            "domain": domain,
            "skill": skill,
            "subskill": subskill,
            "mastery_group": mastery_group,
            "questions_seen": _to_int(row.get("questions_seen", 0)),
            "correct_count": _to_int(row.get("correct_count", 0)),
            "missed_count": _to_int(row.get("missed_count", 0)),
            "incorrect_count": _to_int(row.get("incorrect_count", 0)),
            "accuracy": _to_float(row.get("accuracy", 0.0)),
            "avg_response_time": _to_float(row.get("avg_response_time", 0.0)),
            "recent_accuracy": _to_float(row.get("recent_accuracy", 0.0)),
            "recent_avg_response_time": _to_float(row.get("recent_avg_response_time", 0.0)),
            "current_multiplier": _to_float(row.get("current_multiplier", 1.0)),
            "recommended_action": _clean_str(row.get("recommended_action", "")),
            "last_seen": _clean_str(row.get("last_seen", "")),
        }

        row_values = [row_to_write.get(h, "") for h in headers]
        existing_row_num = row_map.get(key)

        if existing_row_num:
            start_col = 1
            end_col = len(headers)
            cell_range = (
                gspread.utils.rowcol_to_a1(existing_row_num, start_col)
                + ":"
                + gspread.utils.rowcol_to_a1(existing_row_num, end_col)
            )
            ws.update(cell_range, [row_values], value_input_option="USER_ENTERED")
        else:
            append_payload.append(row_values)
            append_keys.append(key)

    if append_payload:
        existing_count = len(ws.get_all_values())
        ws.append_rows(append_payload, value_input_option="USER_ENTERED")
        start_row = existing_count + 1
        for i, key in enumerate(append_keys):
            row_map[key] = start_row + i

    return row_map
# def upsert_numerace_user_profile_rows(rows: list[dict]) -> int:
#     """
#     Upsert profile rows into numerace_user_profile using the natural key:
#     username + domain + skill + subskill + mastery_group
#
#     Returns number of processed rows.
#     """
#     if not rows:
#         return 0
#
#     processed = 0
#     ts_now = datetime.now(timezone.utc).isoformat(timespec="seconds")
#
#     for row in rows:
#         username = _clean_str(row.get("username", ""))
#         domain = _clean_str(row.get("domain", ""))
#         skill = _clean_str(row.get("skill", ""))
#         subskill = _clean_str(row.get("subskill", ""))
#         mastery_group = _clean_str(row.get("mastery_group", ""))
#
#         if not username:
#             continue
#
#         match_dict = {
#             "username": username,
#             "domain": domain,
#             "skill": skill,
#             "subskill": subskill,
#             "mastery_group": mastery_group,
#         }
#
#         row_to_write = {
#             "timestamp_updated": _clean_str(row.get("timestamp_updated", "")) or ts_now,
#             "username": username,
#             "domain": domain,
#             "skill": skill,
#             "subskill": subskill,
#             "mastery_group": mastery_group,
#             "questions_seen": _to_int(row.get("questions_seen", 0)),
#             "correct_count": _to_int(row.get("correct_count", 0)),
#             "missed_count": _to_int(row.get("missed_count", 0)),
#             "incorrect_count": _to_int(row.get("incorrect_count", 0)),
#             "accuracy": _to_float(row.get("accuracy", 0.0)),
#             "avg_response_time": _to_float(row.get("avg_response_time", 0.0)),
#             "recent_accuracy": _to_float(row.get("recent_accuracy", 0.0)),
#             "recent_avg_response_time": _to_float(row.get("recent_avg_response_time", 0.0)),
#             "current_multiplier": _to_float(row.get("current_multiplier", 1.0)),
#             "recommended_action": _clean_str(row.get("recommended_action", "")),
#             "last_seen": _clean_str(row.get("last_seen", "")),
#         }
#
#         upsert_row_by_headers("numerace_user_profile", match_dict, row_to_write)
#         processed += 1
#
#     return processed

def get_numerace_attempt_rows_for_group(
    *,
    username: str,
    domain: str,
    skill: str,
    subskill: str,
    mastery_group: str,
) -> list[dict]:
    rows = get_numerace_attempt_rows(username=username)
    return [
        r for r in rows
        if _clean_str(r.get("domain", "")) == _clean_str(domain)
        and _clean_str(r.get("skill", "")) == _clean_str(skill)
        and _clean_str(r.get("subskill", "")) == _clean_str(subskill)
        and _clean_str(r.get("mastery_group", "")) == _clean_str(mastery_group)
    ]

def append_factoring_round(
    *,
    username: str,
    round_key: str,
    round_id: str,
    game_name: str,
    questions_served: int,
    questions_completed: int,
    correct: int,
    incorrect: int,
    attempts_total: int,
    round_time: float,
    average_response_time: float,
    levels_csv: str = "",
    hints_used_total: int = 0,
    factor_tool_uses_total: int = 0,
    invalid_steps_total: int = 0,
    completed: bool = True,
    notes: str = "",
):
    row = {
        "timestamp": datetime.now().isoformat(),
        "username": username,
        "round_key": round_key,
        "round_id": round_id,
        "game_name": game_name,
        "questions_served": int(questions_served),
        "questions_completed": int(questions_completed),
        "correct": int(correct),
        "incorrect": int(incorrect),
        "attempts_total": int(attempts_total),
        "round_time": float(round_time),
        "average_response_time": float(average_response_time),
        "levels_csv": levels_csv,
        "hints_used_total": int(hints_used_total),
        "factor_tool_uses_total": int(factor_tool_uses_total),
        "invalid_steps_total": int(invalid_steps_total),
        "completed": bool(completed),
        "notes": notes,
    }
    return append_row_by_header_unique("factoring_rounds", row, "round_key")


def append_factoring_attempt(
    *,
    attempt_id: str,
    username: str,
    round_key: str,
    round_id: str,
    question_seq: int,
    level: int,
    question_text: str,
    target_expr: str,
    input_text: str,
    parsed_ok: bool,
    equivalent_to_target: bool,
    is_done: bool,
    is_progress_step: bool,
    invalid_step: bool,
    invalid_reason: str,
    reactive_hint: str,
    attempt_number: int,
    response_time: float,
    hints_used_so_far: int,
    factor_tool_used_count: int,
    steps_count: int,
    current_expr_before: str,
    current_expr_after: str,
):
    row = {
        "timestamp": datetime.now().isoformat(),
        "attempt_id": attempt_id,
        "username": username,
        "round_key": round_key,
        "round_id": round_id,
        "question_seq": int(question_seq),
        "level": int(level),
        "question_text": question_text,
        "target_expr": target_expr,
        "input_text": input_text,
        "parsed_ok": bool(parsed_ok),
        "equivalent_to_target": bool(equivalent_to_target),
        "is_done": bool(is_done),
        "is_progress_step": bool(is_progress_step),
        "invalid_step": bool(invalid_step),
        "invalid_reason": invalid_reason,
        "reactive_hint": reactive_hint,
        "attempt_number": int(attempt_number),
        "response_time": float(response_time),
        "hints_used_so_far": int(hints_used_so_far),
        "factor_tool_used_count": int(factor_tool_used_count),
        "steps_count": int(steps_count),
        "current_expr_before": current_expr_before,
        "current_expr_after": current_expr_after,
    }
    return append_row_by_header_unique("factoring_attempts", row, "attempt_id")

def append_solving_equations_round(
    *,
    username: str,
    round_key: str,
    round_id: str,
    game_name: str,
    questions_served: int,
    questions_completed: int,
    correct: int,
    incorrect: int,
    attempts_total: int,
    round_time: float,
    average_response_time: float,
    levels_csv: str = "",
    hints_used_total: int = 0,
    invalid_steps_total: int = 0,
    completed: bool = True,
    notes: str = "",
):
    row = {
        "timestamp": datetime.now().isoformat(),
        "username": username,
        "round_key": round_key,
        "round_id": round_id,
        "game_name": game_name,
        "questions_served": int(questions_served),
        "questions_completed": int(questions_completed),
        "correct": int(correct),
        "incorrect": int(incorrect),
        "attempts_total": int(attempts_total),
        "round_time": float(round_time),
        "average_response_time": float(average_response_time),
        "levels_csv": levels_csv,
        "hints_used_total": int(hints_used_total),
        "invalid_steps_total": int(invalid_steps_total),
        "completed": bool(completed),
        "notes": notes,
    }
    return append_row_by_header_unique("solving_equations_rounds", row, "round_key")


def append_solving_equations_attempt(
    *,
    attempt_id: str,
    username: str,
    round_key: str,
    round_id: str,
    question_seq: int,
    level: int,
    question_text: str,
    target_solution: str,
    input_mode: str,
    input_text: str,
    parsed_ok: bool,
    equivalent_to_current: bool,
    is_done: bool,
    is_progress_step: bool,
    invalid_step: bool,
    invalid_reason: str,
    reactive_hint: str,
    attempt_number: int,
    response_time: float,
    hints_used_so_far: int,
    steps_count: int,
    current_equation_before: str,
    current_equation_after: str,
):
    row = {
        "timestamp": datetime.now().isoformat(),
        "attempt_id": attempt_id,
        "username": username,
        "round_key": round_key,
        "round_id": round_id,
        "question_seq": int(question_seq),
        "level": int(level),
        "question_text": question_text,
        "target_solution": target_solution,
        "input_mode": input_mode,
        "input_text": input_text,
        "parsed_ok": bool(parsed_ok),
        "equivalent_to_current": bool(equivalent_to_current),
        "is_done": bool(is_done),
        "is_progress_step": bool(is_progress_step),
        "invalid_step": bool(invalid_step),
        "invalid_reason": invalid_reason,
        "reactive_hint": reactive_hint,
        "attempt_number": int(attempt_number),
        "response_time": float(response_time),
        "hints_used_so_far": int(hints_used_so_far),
        "steps_count": int(steps_count),
        "current_equation_before": current_equation_before,
        "current_equation_after": current_equation_after,
    }
    return append_row_by_header_unique("solving_equations_attempts", row, "attempt_id")

def save_pref(username: str, theme: str | None = None, difficulty: str | None = None, last_skill: str | None = None):
    """
    Upsert a preference row in 'prefs' sheet by username.
    Sheet columns expected: username, theme, difficulty, last_skill
    """
    ws = get_sheet("prefs")
    rows = ws.get_all_records()
    for i, r in enumerate(rows, start=2):
        if r.get("username") == username:
            ws.update(f"A{i}:D{i}", [[username, theme, difficulty, last_skill]])
            return
    ws.append_row([username, theme, difficulty, last_skill])


# ---------------------------
# Google Drive helpers
# ---------------------------
def delete_drive_file(file_id: str):
    """
    Delete a Drive file (best effort). Raises GoogleDriveUserFacingError with clear guidance.
    """
    if not file_id:
        raise ValueError("file_id is required")

    try:
        service = get_drive_service()
        service.files().delete(fileId=file_id).execute()
    except Exception as e:
        _raise_drive_user_facing_error(e, context=f"deleting Drive file {file_id}")

def upload_bytes_to_drive(
    *,
    data: bytes,
    filename: str,
    mime_type: str,
    folder_id: str,
) -> str:
    """
    Upload bytes to a Drive folder and return file_id.
    Raises GoogleDriveUserFacingError with clear instructions on failure.
    """
    if not folder_id:
        raise ValueError("folder_id is required (raw Drive folder id)")

    try:
        service = get_drive_service()
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type, resumable=True)
        file_metadata = {"name": filename, "parents": [folder_id]}
        created = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
        return created["id"]
    except Exception as e:
        _raise_drive_user_facing_error(e, context=f"uploading '{filename}' to folder {folder_id}")

def download_drive_file_bytes(file_id: str) -> bytes:
    """Download Drive file into memory, return bytes. Raises GoogleDriveUserFacingError on failure."""
    if not file_id:
        raise ValueError("file_id is required")

    try:
        service = get_drive_service()
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

        return fh.getvalue()
    except Exception as e:
        _raise_drive_user_facing_error(e, context=f"downloading Drive file {file_id}")

def set_drive_file_public_read(file_id: str):
    """Best-effort: make file readable by anyone with the link."""
    try:
        service = get_drive_service()
        service.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
            fields="id",
        ).execute()
    except GoogleDriveUserFacingError:
        # If Drive auth is broken, surface it (publishing can't continue reliably)
        raise
    except Exception:
        # If the account disallows public link sharing, don't fail the publish.
        pass

def upload_interactive_json(
    *,
    obj: dict,
    filename: str | None = None,
    title: str | None = None,
    make_public: bool | None = None,
) -> dict[str, Any]:
    """
    Upload JSON worksheet to the Interactives folder and return a standard payload
    to store in PublishedItems.content.

    Provide either `filename` or `title` (title -> safe_filename).
    """
    cfg = _get_google_config()
    folder_id = cfg.interactives_folder_id
    if not folder_id:
        raise ValueError("Missing gdrive.interactives_folder_id in secrets.toml")

    if make_public is None:
        make_public = cfg.default_make_public

    if not filename:
        filename = safe_filename(title or "interactive", ".json")

    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    file_id = upload_bytes_to_drive(
        data=data,
        filename=filename,
        mime_type="application/json",
        folder_id=folder_id,
    )

    if make_public:
        set_drive_file_public_read(file_id)

    urls = gdrive_urls(file_id)
    return {
        "provider": "gdrive",
        "folder": "Interactives",
        "file_id": file_id,
        "filename": filename,
        "view_url": urls["view_url"],
        "download_url": urls["download_url"],
    }


def upload_pdf_bytes(
    *,
    pdf_bytes: bytes,
    filename: str | None = None,
    title: str | None = None,
    make_public: bool | None = None,
) -> dict[str, Any]:
    """
    Upload PDF bytes to the PDFs folder and return standard payload for PublishedItems.content.

    Provide either `filename` or `title` (title -> safe_filename).
    """
    cfg = _get_google_config()
    folder_id = cfg.pdfs_folder_id
    if not folder_id:
        raise ValueError("Missing gdrive.pdfs_folder_id in secrets.toml")

    if make_public is None:
        make_public = cfg.default_make_public

    if not filename:
        filename = safe_filename(title or "notes", ".pdf")

    file_id = upload_bytes_to_drive(
        data=pdf_bytes,
        filename=filename,
        mime_type="application/pdf",
        folder_id=folder_id,
    )

    if make_public:
        set_drive_file_public_read(file_id)

    urls = gdrive_urls(file_id)
    return {
        "provider": "gdrive",
        "folder": "PDFs",
        "file_id": file_id,
        "filename": filename,
        "view_url": urls["view_url"],
        "preview_url": urls["preview_url"],
        "download_url": urls["download_url"],
    }

def update_drive_file_bytes(*, file_id: str, data: bytes, mime_type: str):
    """
    Replace the contents of an existing Drive file (same file_id).
    Useful for editing interactive JSON in place.
    """
    if not file_id:
        raise ValueError("file_id is required")

    try:
        service = get_drive_service()
        media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type, resumable=True)
        service.files().update(fileId=file_id, media_body=media).execute()
    except Exception as e:
        _raise_drive_user_facing_error(e, context=f"updating Drive file {file_id}")
