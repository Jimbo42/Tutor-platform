import sys
from pathlib import Path
import base64

ROOT = Path(__file__).resolve().parents[1]   # points to Tutor/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from shared.published_db import init_published_db
from shared.auth import verify_password
#from shared.user_data import init_user_data_db

def load_image_base64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode()

st.set_page_config(page_title="TutorAssist", page_icon="🦾", layout="wide")

BG_IMAGE = Path(__file__).parent / "resources" / "images" / "background.jpg"
bg_base64 = load_image_base64(BG_IMAGE)


st.markdown(
    f"""
    <style>
    /* Main app container */
    [data-testid="stAppViewContainer"] {{
        position: relative;
    }}

    /* Background layer */
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        inset: 0;
        background-image: url("data:image/jpg;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        filter: blur(1px) brightness(0.75);
        opacity: 0.25;           /* 👈 control fade here */
        z-index: 0;
        pointer-events: none;
    }}

    /* Make sure all content stays above */
    [data-testid="stAppViewContainer"] > * {{
        position: relative;
        z-index: 1;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Init shared DBs
# ----------------------------
init_published_db()
#init_user_data_db()

# ----------------------------
# Auth helpers
# ----------------------------
def logout():
    st.session_state.authenticated = False
    st.session_state.username = None

def require_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None

    if st.session_state.authenticated:
        return

    st.title("🔐 TutorAssist Login")

    # Load credentials dict from secrets
    users = st.secrets.get("APP_PASSWORDS", {})
    if not users:
        st.error("No APP_PASSWORDS found in secrets.")
        st.stop()

    u = st.text_input("Username")
    pw = st.text_input("Password", type="password")

    col1, col2 = st.columns([1, 1])
    with col1:
        login_clicked = st.button("Login", type="primary", use_container_width=True)
    with col2:
        st.caption("")

    if login_clicked:
        stored = users.get(u)
        if stored and verify_password(pw, stored):
            st.session_state.authenticated = True
            st.session_state.username = u
            st.toast(f"Welcome, {u}!", icon="✅")
            st.rerun()
        else:
            st.error("Incorrect username or password")

    st.stop()

# ----------------------------
# Gate everything behind login
# ----------------------------
require_login()

# -------------------------------------------------
# Sidebar: Logout
# -------------------------------------------------
with st.sidebar:
    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.rerun()

pages = {
    "Home": [
        st.Page("home.py", title="Home", default=True),
    ],
    "Math Skills Challenges": [
        st.Page("math_factoring.py", title="Factoring"),
        st.Page("math_numeracy.py", title="NumerAce"),
    ],
    "Library": [
        st.Page("practice.py", title="Practice Questions"),
    ],
    "Resources": [
        st.Page("notes.py", title="Notes"),
        st.Page("formula_a.py", title="Formula List"),
    ],
    "Tools": [
        st.Page("chemistry_a.py", title="Chemistry Calculator"),
    ],
}

init_published_db()

pg = st.navigation(pages)
pg.run()

