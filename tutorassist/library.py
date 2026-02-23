import streamlit as st

from practice import show_practice_library
from notes import show_notes_library

# Begin rendering
st.markdown("""
<style>
/* overall page top padding */
div.block-container { padding-top: 0.6rem !important; }

/* DO NOT shift tabs upward (no negative margins) */
div[data-testid="stTabs"] { margin-top: 0.2rem !important; }

/* tighten headings */
h1, h2, h3 { margin-top: 0.2rem !important; margin-bottom: 0.4rem !important; }

/* tighten markdown paragraphs/captions */
div[data-testid="stMarkdownContainer"] p { margin-bottom: 0.35rem !important; }

/* tighten selectbox labels spacing */
label { margin-bottom: 0.15rem !important; }

/* reduce extra space above first element */
section.main > div { padding-top: 0rem !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📝 Practice", "📘 Notes"])

with tab1:
    show_practice_library()

with tab2:
    show_notes_library()
