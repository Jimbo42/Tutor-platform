import streamlit as st
from streamlit import session_state as ss

from tutortrack.numerace_reports import show_numerace_reports
from tutortrack.numerace_profile_admin import show_numerace_profile_admin
from tutortrack.factoring_reports import show_factoring_reports
from tutortrack.solving_equations_reports import show_solving_equations_reports

REPORT_ROUTES = {
    "NumeRace": show_numerace_reports,
    "Numerace Profiles": show_numerace_profile_admin,
    "Factoring": show_factoring_reports,
    "Solving Equations": show_solving_equations_reports
}
# Begin rendering
st.markdown(
    """
    <style>
    /* Make main container wider */
    .block-container {
        padding-top: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 98%;
    }

    /* Make headers bigger */
    h1, h2, h3 {
        letter-spacing: 0.5px;
    }

    /* Make data editor wider */
    [data-testid="stDataFrame"] {
        width: 100%;
    }

    /* Slightly bigger text in tables */
    [data-testid="stDataFrame"] div {
        font-size: 15px;
    }

    /* Buttons slightly larger */
    button {
        font-size: 15px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("📋 Reporting")
with col2:
    title = st.empty()

if "reportMode" not in ss:
    ss.reportMode = None

with st.sidebar:
    for name in REPORT_ROUTES:
        if st.button(name, width="stretch"):
            ss.reportMode = name

if ss.reportMode in REPORT_ROUTES:
    REPORT_ROUTES[ss.reportMode]()
else:
    st.info("Select a report from the sidebar.")
