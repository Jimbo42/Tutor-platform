import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime


# ---------------------------
# CONNECT TO GOOGLE SHEETS
# ---------------------------
def get_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",       # <-- add this
    ]
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes
    )
    return gspread.authorize(creds)


def get_sheet(sheet_name):
    client = get_client()
    sh = client.open_by_key(st.secrets["gSheets"]["spreadsheet_id"])

    return sh.worksheet(sheet_name)


# ---------------------------
# SCORES
# ---------------------------
def append_score(username, skill, quiz_id, score, max_score, details=None):
    ws = get_sheet("scores")

    ws.append_row([
        datetime.now().isoformat(),
        username,
        skill,
        quiz_id,
        score,
        max_score,
        details or ""
    ])

from datetime import datetime

def append_numerace_round(
    username: str,
    total_questions: int,
    incorrect: int,
    missed: int,
    attempts_total: int,
    round_time: float,
    average_response_time: float
):
    """
    Append one NumeRace round summary row into worksheet 'numerace'.
    """
    ws = get_sheet("numerace")  # Make sure the tab exists in the spreadsheet

    ws.append_row([
        datetime.now().isoformat(),
        username,
        int(total_questions),
        int(incorrect),
        int(missed),
        int(attempts_total),
        float(round_time),
        float(average_response_time),
    ])

# ---------------------------
# USER PREFERENCES
# ---------------------------
def save_pref(username, theme=None, difficulty=None, last_skill=None):
    ws = get_sheet("prefs")

    rows = ws.get_all_records()

    for i, r in enumerate(rows, start=2):
        if r["username"] == username:
            ws.update(
                f"A{i}:D{i}",
                [[username, theme, difficulty, last_skill]]
            )
            return

    ws.append_row([username, theme, difficulty, last_skill])
