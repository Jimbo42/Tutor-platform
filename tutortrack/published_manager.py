import streamlit as st
import pandas as pd
from shared.published_db import (
    list_published_items,
    get_published_item_full,
    update_published_item,
    delete_published_item
)
from shared.content_renderer import render_published_content

@st.dialog("👁 Preview Published Item", width="large")
def open_preview_dialog(item_id):
    item = get_published_item_full(item_id)

    if not item:
        st.error("Item not found.")
        return

    _, title, subject, grade, ctype, content, created, updated, visible = item

    st.subheader(title)
    st.caption(f"{subject} • Grade {grade} • {ctype}")

    st.divider()

    render_published_content(content)

@st.dialog("✏️ Edit Published Item")
def open_edit_dialog(item_id):
    item = get_published_item_full(item_id)

    if not item:
        st.error("Item not found.")
        return

    _, title, subject, grade, ctype, content, created, updated, visible = item

    with st.form("edit_published"):
        title = st.text_input("Title", title)
        subject = st.text_input("Subject", subject)
        grade = st.text_input("Grade", grade)
        ctype = st.selectbox("Type", ["questions", "notes", "practice", "explanation"], index=0)
        visible = st.checkbox("Visible", bool(visible))

        st.markdown("### Content")
        content = st.text_area("JSON or Markdown", content, height=300)

        if st.form_submit_button("💾 Save Changes"):
            update_published_item(item_id, title, subject, grade, ctype, content, visible)
            st.toast("Saved", icon="✅")
            st.rerun()

@st.dialog("🗑 Delete Published Item")
def open_delete_dialog(item_id):
    st.warning("This cannot be undone.")

    if st.button("❌ Confirm Delete"):
        delete_published_item(item_id)
        st.toast("Deleted", icon="🗑️")
        st.rerun()

# Render page
def show_published_manager():

    st.header("📦 Published Content Manager")

    rows = list_published_items()

    if not rows:
        st.info("No published items yet.")
        st.stop()

    df = pd.DataFrame(rows, columns=[
        "id", "title", "subject", "grade", "type", "created", "updated", "visible"
    ])

    df.insert(0, "Preview", False)
    df.insert(1, "Edit", False)
    df.insert(1, "Delete", False)

    # Display table
    edited = st.data_editor(
        df,
        hide_index=True,
        height=500,
        column_config={
            "Preview": st.column_config.CheckboxColumn("Preview"),
            "Edit": st.column_config.CheckboxColumn("Edit"),
            "Delete": st.column_config.CheckboxColumn("Delete"),
            "id": st.column_config.NumberColumn("ID"),
        }
    )

    # Handle actions
    for i, row in edited.iterrows():
        item_id = int(row["id"])

        if row["Preview"]:
            open_preview_dialog(item_id)

        if row["Edit"]:
            open_edit_dialog(item_id)

        if row["Delete"]:
            open_delete_dialog(item_id)
