import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # points to Tutor/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from shared.published_db import init_published_db


def load_image_base64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode()

st.set_page_config(page_title="TutorTrack", page_icon="👨‍🏫", layout="wide")

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

pages = {
    "Lessons": [
        st.Page("lessons.py", title="Lessons"),
        st.Page("written.py", title="Written Notes"),
    ],
    "Resources": [
        st.Page("resources.py", title="Resource List"),
        st.Page("prompts.py", title="Prompts"),
    ],
    "Tools": [
        st.Page("chemistry.py", title="Chemistry"),
        st.Page("configuration.py", title="Configuration"),
    ],
}

init_published_db()

pg = st.navigation(pages)
pg.run()

