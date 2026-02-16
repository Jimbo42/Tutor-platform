import streamlit as st
import pandas as pd
from streamlit import session_state as ss

from shared.published_db import (
    list_published_items,
    get_published_item_full,
    update_published_item,
    delete_published_item
)

from shared.content_renderer import render_published_content


# -----------------------------
# Dialogs
# -----------------------------
@st.dialog("👁 Preview Published Item", width="large")
def open_preview_dialog(item_id: int):
    item = get_published_item_full(item_id)
    if not item:
        st.error("Item not found.")
        return

    _, title, subject, grade, ctype, content, created, updated, visible = item

    st.subheader(title)
    st.caption(f"{subject} • Grade {grade} • {ctype}")
    st.divider()

    render_published_content(content, content_type=ctype)

    st.divider()
    if st.button("Close", width="stretch"):
        st.rerun()


@st.dialog("✏️ Edit Published Item")
def open_edit_dialog(item_id: int):
    item = get_published_item_full(item_id)
    if not item:
        st.error("Item not found.")
        return

    _, title, subject, grade, ctype, content, created, updated, visible = item

    with st.form("edit_published"):
        title = st.text_input("Title", title)
        subject = st.text_input("Subject", subject)
        grade = st.text_input("Grade", grade)
        ctype = st.text_input("Type", ctype)  # keep flexible
        visible = st.checkbox("Visible", bool(visible))

        st.markdown("### Content")
        content = st.text_area("JSON or Markdown", content, height=300)

        if st.form_submit_button("💾 Save Changes"):
            update_published_item(item_id, title, subject, grade, ctype, content, visible)
            st.toast("Saved", icon="✅")
            st.rerun()


@st.dialog("🗑 Delete Published Item")
def open_delete_dialog(item_id: int):
    item = get_published_item_full(item_id)
    if not item:
        st.error("Item not found.")
        return

    _, title, subject, grade, ctype, content, created, updated, visible = item
    st.warning(f"Delete **{title}**? This cannot be undone.")

    if st.button("❌ Confirm Delete", width="stretch"):
        delete_published_item(item_id)
        st.toast("Deleted", icon="🗑️")
        st.rerun()


# -----------------------------
# Page
# -----------------------------
def show_published_manager():
    st.header("📦 Published Content Manager")

    rows = list_published_items()
    if not rows:
        st.info("No published items yet.")
        st.stop()

    df = pd.DataFrame(rows, columns=[
        "id", "title", "subject", "grade", "type", "created", "updated", "visible"
    ])

    # ---- ACTION HANDLER ----
    def handle_action(row_idx: int):
        key = f"pub_action_{row_idx}"
        choice = ss.get(key, "·")

        item_id = int(df.iloc[row_idx]["id"])

        if choice == "👁":
            open_preview_dialog(item_id)

        elif choice == "✏️":
            open_edit_dialog(item_id)

        elif choice == "🗑️":
            open_delete_dialog(item_id)

        # reset pill back to neutral so it doesn't re-trigger
        ss[key] = "·"

    st.markdown("""
    <style>
    div[data-testid="stButtonGroup"] button:first-child { display:none; }
    </style>
    """, unsafe_allow_html=True)

    # ---- LIST ----
    with st.expander("📚 Published Items", expanded=True):
        for i, row in df.iterrows():
            label = f"{row['title']} — {row['subject']} • Grade {row['grade']} • {row['type']}"
            st.pills(
                label="Published item action",
                label_visibility="collapsed",
                options=["·", "👁", "✏️", "🗑️", label],
                key=f"pub_action_{i}",
                width="content",
                on_change=handle_action,
                args=(i,),
            )
