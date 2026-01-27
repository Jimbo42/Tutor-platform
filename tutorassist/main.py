import sys
from pathlib import Path
import base64

ROOT = Path(__file__).resolve().parents[1]   # points to Tutor/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from shared.published_db import init_published_db

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

# -------------------------------
# 🔐 Simple password protection
# -------------------------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:

    st.title("🔐 TutorAssist Login")

    # Centered, nicer login box
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        with st.container(border=True):
            st.subheader("Please sign in")

            pw = st.text_input("Password", type="password")

            if st.button("Login", use_container_width=True):
                if pw == st.secrets["APP_PASSWORD"]:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password")

    st.stop()

# -------------------------------------------------
# Sidebar: Logout
# -------------------------------------------------
with st.sidebar:
    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.rerun()

pages = {
    "Math Skills Challenges": [
        st.Page("math_skills.py", title="Factoring"),
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

