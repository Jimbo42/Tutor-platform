import sqlite3
import streamlit as st
from streamlit import session_state as ss
import pandas as pd
import qrcode
from pathlib import Path
import re

from shared.formulas import show_formulas
from published_manager import show_published_manager
from tutortrack.lessons import get_conn

_DRIVE_FILE_ID = re.compile(r"/file/d/([a-zA-Z0-9_-]+)")
_DRIVE_UC_ID   = re.compile(r"[?&]id=([a-zA-Z0-9_-]+)")
_DRIVE_OPEN_ID = re.compile(r"/open\?id=([a-zA-Z0-9_-]+)")

def extract_gdrive_file_id(url_or_id: str) -> str | None:
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

    # allow raw id
    if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", s):
        return s

    return None

def gdrive_urls(file_id: str) -> dict:
    return {
        "view_url": f"https://drive.google.com/file/d/{file_id}/view",
        "preview_url": f"https://drive.google.com/file/d/{file_id}/preview",
        "download_url": f"https://drive.google.com/uc?export=download&id={file_id}",
    }


BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
IMG_PATH = BASE_DIR / "resources" / "images"

def get_resource_list():
    conn_l = get_conn()
    cl = conn_l.cursor()
    with conn_l:
        cl.execute("SELECT Name, Subject, Link, Rating, Themes, Tags, ROWID FROM Resources ORDER BY Name")
        resources = cl.fetchall()
        # Create a DataFrame with column names
        ss.resources = pd.DataFrame(resources, columns=['Name', 'Subject', 'Link', 'Rating', 'Themes', 'Tags', 'RowID'])
        ss.resources.insert(0,"Edit", False)
        ss.resources.insert(1,"QR_Code", False)

    conn_l.close()

def get_tags_themes():
    conn_r = get_conn()
    cr = conn_r.cursor()
    with conn_r:
        cr.execute("SELECT Tag FROM Tags")
        tags = cr.fetchall()
        ss.tags = pd.DataFrame(tags, columns=['Tag'])

        cr.execute("SELECT Theme FROM Themes")
        themes = cr.fetchall()
        ss.themes = pd.DataFrame(themes, columns=['Theme'])

    conn_r.close()

#  Dialog Boxes
@st.dialog("Add Resource")
def add_new_resource():
    with st.form("Resource"):
        resource_name = st.text_input("Resource Name")
        subject = st.text_input("Subject")
        link = st.text_input("Link")
        rating = st.number_input("Rating")
        Tags = st.multiselect("Tags", ss.tags)
        Themes = st.multiselect("Themes", ss.themes)

        if st.form_submit_button("Save Resource"):
            tag_list = ", ".join(Tags)
            theme_list = ", ".join(Themes)
            conn_a = get_conn()
            ca = conn_a.cursor()
            with conn_a:
                ca.execute(
                    "INSERT INTO Resources (Name, Subject, Link, Rating, Tags, Themes)"
                    " VALUES (:resource, :subject, :link, :rating, :themes, :tags)",
                    {"resource": resource_name, "subject": subject, "link": link, "rating": rating, "tags": tag_list, "themes": theme_list})

            conn_a.close()
            get_resource_list()
            st.rerun()

def get_resource_details():
    if ss.resource_row:
        edited = ss.resource_row['edited_rows']
        if edited:
            row_num = next(iter(edited))
            row_data = edited[row_num]
            if row_data.get('QR_Code'):
                generate_qr_code(ss.resources.iloc[row_num].Link)
            elif row_data.get('Edit'):
                edit_resource(row_num)

@st.dialog("Edit Resource")
def edit_resource(rowNum):
    rowID = int(ss.resources.iloc[rowNum].RowID)
    rValue = float(ss.resources.iloc[rowNum].Rating) if ss.resources.iloc[rowNum].Rating else 0.0
    with st.form("Resource"):
        # Input fields pre-filled with current values
        resource_name = st.text_input("Resource Name", ss.resources.iloc[rowNum].Name)
        subject = st.text_input("Subject", ss.resources.iloc[rowNum].Subject)
        link = st.text_input("Link", ss.resources.iloc[rowNum].Link)
        ratingValue = st.number_input("Rating", rValue)
        Tags = st.multiselect("Tags", ss.tags, ss.resources.iloc[rowNum].Tags)
        Themes = st.multiselect("Themes", ss.themes, ss.resources.iloc[rowNum].Themes)

        if st.form_submit_button("Update Resource"):
            # Convert Tags and Themes list to comma-separated strings
            tag_list = ", ".join(Tags)
            theme_list = ", ".join(Themes)
            rating = str(ratingValue)  # Convert rating to string if needed

            # Now you can update the table (Resources or Last_Request)
            try:
                conn_a = get_conn()
                ca = conn_a.cursor()

                # Assuming you want to update the Last_Request table
                ca.execute(
                    """
                    UPDATE Resources
                    SET Name = :resource, Subject = :subject, Link = :link, Rating = :rating,
                        Tags = :tags, Themes = :themes
                    WHERE ROWID = :rowID
                    """,
                    {
                        "resource": resource_name,
                        "subject": subject,
                        "link": link,
                        "rating": rating,  # Rating as a string or float
                        "tags": tag_list,  # Tags as comma-separated string
                        "themes": theme_list,  # Themes as comma-separated string
                        "rowID": rowID
                    }
                )

                conn_a.commit()  # Commit the transaction
                if ca.rowcount > 0:
                    st.success("Resource updated successfully!")
                else:
                    st.warning(f"Could not update for ROWID= {rowID}")

            except sqlite3.Error as e:
                st.error(f"An error occurred: {e}")

            finally:
                conn_a.close()  # Ensure the connection is closed

    get_resource_list()
    st.rerun()

def generate_qr_code(text):
    file_path = IMG_PATH / "qr_code.png"
    img = qrcode.make(text)
    img.save(file_path)
    with st.popover("QR Code"):
        st.image(file_path)

# Render screen
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
    /* Make header rows align nicely */
    [data-testid="stHorizontalBlock"] {
    align-items: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)

if "ResourceMode" not in ss:
    ss.ResourceMode = "Resources"

#headline = st.header(ss.ResourceMode)
with st.sidebar:
    if st.button("Online ☁️"):
        ss.ResourceMode = "Online"
    if st.button("Formulas 🧠"):
        ss.ResourceMode= "Formulas"
    if st.button("Published Manager "):
        ss.ResourceMode= "Published"

if ss.ResourceMode == "Online":

    if 'tags' not in ss:
        ss.tags = pd.DataFrame([], columns=['Tag'])
        ss.themes = pd.DataFrame([], columns=['Theme'])

    col1, col2 = st.columns([8, 1])

    with col1:
        st.header("📚 Resources")
    with col2:
        st.markdown("<div style='height: 3.2rem'></div>", unsafe_allow_html=True)
        if st.button("➕", key="Resource_editor", help="Add New Resource"):
            add_new_resource()

    if 'resources' not in ss:
        get_resource_list()
        get_tags_themes()

    column_config = {
        "Edit": st.column_config.CheckboxColumn(
            "Edit"
        ),
        "QR Code": st.column_config.CheckboxColumn(
            "QR_Code"
        ),
        "Name": st.column_config.TextColumn(
            "Resource"
        ),
        "Subject": st.column_config.TextColumn(
            "Subject"
        ),
        "Link": st.column_config.LinkColumn(
            "Link"
        ),
        "Rating": st.column_config.TextColumn(
            "Rating"
        ),
        "Themes": st.column_config.ListColumn(
            "Themes"
        ),
        "Tags": st.column_config.ListColumn(
            "Tags"
        ),
        "RowID": st.column_config.NumberColumn(
            "RowID"
        )
    }

    search_query = st.text_input("🔎 Search resources", "", placeholder="Search by tag, theme, or subject...")

    if search_query:
    #    filtered_resources = ss.resources[ss.resources['Tags'].apply( lambda tags: any(search_query.lower() in tag.lower() for tag in tags))]
        filtered_resources = ss.resources[ss.resources['Tags'].str.contains( search_query, case=False, na=False)
            | ss.resources['Themes'].str.contains( search_query, case=False, na=False)
            | ss.resources['Subject'].str.contains( search_query, case=False, na=False)
        ]
    else:
        filtered_resources = ss.resources

    st.data_editor(filtered_resources,
                 column_config=column_config,
                 on_change=get_resource_details,
                 key="resource_row",
                 hide_index=True,
                 height=650,
                 width="stretch")

if ss.ResourceMode == "Formulas":
    show_formulas()

if ss.ResourceMode == "Published":
    show_published_manager()