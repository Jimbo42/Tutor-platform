import sqlite3
import streamlit as st
from streamlit import session_state as ss
import pandas as pd
import os
import qrcode
from pathlib import Path

from shared.formulas import show_formulas
from published_manager import show_published_manager
from tutortrack.lessons import get_conn

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
IMG_PATH = BASE_DIR / "resources" / "images"
PDF_DIR = PARENT_DIR / "shared" / "pdf_files"
PDF_DIR.mkdir(parents=True, exist_ok=True)
pdf_folder = str(PDF_DIR)

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

@st.dialog("Rename PDF")
def rename_pdf_dialog(old_name: str, pdf_folder: str):
    st.write(f"Current file name:\n**{old_name}**")

    new_name = st.text_input("New name", value=old_name, key="rename_input")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Cancel"):
            ss.pop("rename_target", None)
            st.rerun()

    with col2:
        if st.button("Confirm Rename"):
            old_path = os.path.join(pdf_folder, old_name)
            new_path = os.path.join(pdf_folder, new_name)

            if old_name != new_name and os.path.exists(old_path):
                os.rename(old_path, new_path)

            ss.pop("rename_target", None)
            st.rerun()


def resources_pdf_viewer():

    st.markdown("""
    <style>
    main {
    padding-top: 0rem !important;
    margin-top: 0rem !important;
    }
    /* Reduce top padding of main app container */
    div.block-container {
        padding-top: 0.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.header("Saved PDF Resources")

    os.makedirs(pdf_folder, exist_ok=True)

    pdf_files = sorted(
        f for f in os.listdir(pdf_folder)
        if f.lower().endswith(".pdf")
    )

    if not pdf_files:
        st.info("No PDF files found in saved_files folder.")
        return

    # ---- ACTION HANDLER ----
    def handle_action(row):
        key = f"pdf_action_{row}"
        choice = ss[key]
        filename = pdf_files[row]
        full_path = os.path.join(pdf_folder, filename)

        if choice == "👀":
            ss["selected_pdf"] = filename

        elif choice == "🗑️":
            os.remove(full_path)
            st.rerun()

        elif choice == "✏️":
            ss["rename_target"] = filename

        # reset selection back to neutral
        ss[key] = "·"

    st.markdown("""
    <style>
    div[data-testid="stButtonGroup"] button:first-child {
        display:none;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---- FILE LIST (auto expand / collapse) ----
    expanded = "selected_pdf" not in ss

    with st.expander("📂 PDF Files", expanded=expanded):
        for i, filename in enumerate(pdf_files):
            st.pills(
                label="PDF file action",
                label_visibility="collapsed",
                options=["·", "👀", "🗑️", "✏️", filename],
                key=f"pdf_action_{i}",
                width="content",
                on_change=handle_action,
                args=(i,),
            )

    if "selected_pdf" in ss:
        filename = ss["selected_pdf"]
        st.subheader(filename)
        st.pdf(os.path.join(pdf_folder, filename))

    # ---- RENAME ----
    if "rename_target" in ss:
        old = ss["rename_target"]
        rename_pdf_dialog(old, pdf_folder)

def generate_qr_code(text):
    file_path = IMG_PATH / "qr_code.png"
    img = qrcode.make(text)
    img.save(file_path)
    with st.popover("QR Code"):
        st.image(file_path)

if "ResourceMode" not in ss:
    ss.ResourceMode = "Resources"

#headline = st.header(ss.ResourceMode)
with st.sidebar:
    if st.button("Online ☁️"):
        ss.ResourceMode = "Online"
    if st.button("Documents 📄"):
        ss.ResourceMode = "Documents"
    if st.button("Formulas 🧠"):
        ss.ResourceMode= "Formulas"
    if st.button("Published Manager "):
        ss.ResourceMode= "Published"

if ss.ResourceMode == "Online":

    if 'tags' not in ss:
        ss.tags = pd.DataFrame([], columns=['Tag'])
        ss.themes = pd.DataFrame([], columns=['Theme'])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Resources")
    with col2:
        if st.button("➕", key=f"Resource_editor", help="Add New Resource"):
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

    search_query = st.text_input("Search:", "")
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
                 hide_index=True, height=600)

if ss.ResourceMode == "Documents":
    resources_pdf_viewer()

if ss.ResourceMode == "Formulas":
    show_formulas()

if ss.ResourceMode == "Published":
    show_published_manager()