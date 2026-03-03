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
from datetime import datetime
from typing import Any, Optional

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
    headers = ws.row_values(1)
    if not headers:
        raise ValueError(f"Sheet '{tab_name}' has no header row (row 1).")

    row = [row_dict.get(h, "") for h in headers]

    # USER_ENTERED lets TRUE/FALSE, dates, etc behave nicely
    ws.append_row(row, value_input_option="USER_ENTERED")


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

# def get_drive_service():
#     """Cached Drive service client (Drive API v3). Prefer user OAuth for My Drive uploads."""
#     if "drive_service" in st.session_state:
#         return st.session_state["drive_service"]
#
#     # ✅ Use OAuth user creds if present (personal My Drive quota)
#     if "gdrive_oauth" in st.secrets:
#         cfg = st.secrets["gdrive_oauth"]
#         creds = UserCredentials(
#             token=None,
#             refresh_token=cfg["refresh_token"],
#             token_uri=cfg.get("token_uri", "https://oauth2.googleapis.com/token"),
#             client_id=cfg["client_id"],
#             client_secret=cfg["client_secret"],
#             scopes=["https://www.googleapis.com/auth/drive.file"],
#         )
#         creds.refresh(Request())
#         st.session_state["drive_service"] = build(
#             "drive", "v3", credentials=creds, cache_discovery=False
#         )
#         return st.session_state["drive_service"]
#
#     # Fallback: service account (uploads to My Drive will fail with quota error)
#     st.session_state["drive_service"] = build(
#         "drive",
#         "v3",
#         credentials=get_credentials(scopes=["https://www.googleapis.com/auth/drive"]),
#         cache_discovery=False,
#     )
#     return st.session_state["drive_service"]

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
    Return a gspread Worksheet by name from the configured spreadsheet.
    """
    cfg = _get_google_config()
    if not cfg.spreadsheet_id:
        raise ValueError("Missing gSheets.spreadsheet_id in secrets.toml")

    client = get_gspread_client()
    sh = client.open_by_key(cfg.spreadsheet_id)
    return sh.worksheet(sheet_name)


def append_row(sheet_name: str, values: list[Any]):
    ws = get_sheet(sheet_name)
    ws.append_row(values)


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


def append_numerace_round(
    username: str,
    total_questions: int,
    incorrect: int,
    missed: int,
    attempts_total: int,
    round_time: float,
    average_response_time: float,
):
    append_row(
        "numerace",
        [
            datetime.now().isoformat(),
            username,
            int(total_questions),
            int(incorrect),
            int(missed),
            int(attempts_total),
            float(round_time),
            float(average_response_time),
        ],
    )


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
# def upload_bytes_to_drive(
#     *,
#     data: bytes,
#     filename: str,
#     mime_type: str,
#     folder_id: str,
# ) -> str:
#     """
#     Upload bytes to a Drive folder and return file_id.
#     """
#     if not folder_id:
#         raise ValueError("folder_id is required (raw Drive folder id)")
#
#     service = get_drive_service()
#     media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type, resumable=True)
#
#     file_metadata = {"name": filename, "parents": [folder_id]}
#     created = service.files().create(
#         body=file_metadata,
#         media_body=media,
#         fields="id",
#     ).execute()
#
#     return created["id"]

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

# def set_drive_file_public_read(file_id: str):
#     """Best-effort: make file readable by anyone with the link."""
#     service = get_drive_service()
#     try:
#         service.permissions().create(
#             fileId=file_id,
#             body={"type": "anyone", "role": "reader"},
#             fields="id",
#         ).execute()
#     except Exception:
#         # If the account disallows public link sharing, don't fail the publish.
#         pass

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

# def update_drive_file_bytes(*, file_id: str, data: bytes, mime_type: str):
#     """
#     Replace the contents of an existing Drive file (same file_id).
#     Useful for editing interactive JSON in place.
#     """
#     if not file_id:
#         raise ValueError("file_id is required")
#
#     service = get_drive_service()
#     media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime_type, resumable=True)
#
#     service.files().update(
#         fileId=file_id,
#         media_body=media,
#     ).execute()

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