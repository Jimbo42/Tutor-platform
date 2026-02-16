import streamlit as st

from shared.content_renderer import (
    render_interactive_questions,
    translate_latex,
)
from shared.published_db import get_published_items, get_published_item_by_id
import json
import requests

@st.cache_data(show_spinner="Loading worksheet from Google Drive...")
def load_interactive_from_gdrive(download_url: str):
    try:
        r = requests.get(download_url, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}

st.title("📚 Practice Library")

# ----------------------------
# Load items
# ----------------------------
items = get_published_items()

if not items:
    st.info("No published items yet.")
    st.stop()

# items rows:
# (id, title, subject, grade, content_type, created_at)

# Convert to list of dicts for easier handling
records = []
for r in items:
    records.append({
        "id": r[0],
        "title": r[1],
        "subject": r[2] or "Other",
        "grade": r[3] or "Other",
        "type": r[4] or "Other",
        "created": r[5],
    })

# ----------------------------
# Filters
# ----------------------------
st.subheader("🔎 Filter")

subjects = sorted(set(r["subject"] for r in records))
grades   = sorted(set(r["grade"] for r in records))
types    = sorted(set(r["type"] for r in records))

c1, c2, c3 = st.columns(3)

with c1:
    f_subject = st.selectbox("Subject", ["All"] + subjects)
with c2:
    f_grade = st.selectbox("Grade", ["All"] + grades)
with c3:
    f_type = st.selectbox("Type", ["All"] + types)

def passes_filters(r):
    if f_subject != "All" and r["subject"] != f_subject:
        return False
    if f_grade != "All" and r["grade"] != f_grade:
        return False
    if f_type != "All" and r["type"] != f_type:
        return False
    return True

filtered = [r for r in records if passes_filters(r)]

if not filtered:
    st.warning("No items match these filters.")
    st.stop()

st.divider()

# ----------------------------
# Direct link support
# ----------------------------
qp = st.query_params
forced_id = qp.get("item")

# ----------------------------
# Build selector
# ----------------------------
label_map = {}
labels = []

for r in filtered:
    label = f"{r['title']}  —  {r['subject']} • Grade {r['grade']} • {r['type']}"
    labels.append(label)
    label_map[label] = r["id"]

# Pick default
default_index = 0
if forced_id:
    try:
        forced_id = int(forced_id)
        for i, r in enumerate(filtered):
            if r["id"] == forced_id:
                default_index = i
                break
    except:
        pass

# Add placeholder option
labels_with_placeholder = ["— Select an item —"] + labels

# If URL forced_id exists, try to preselect it
default_index = 0

if forced_id:
    try:
        forced_id = int(forced_id)
        for i, r in enumerate(filtered):
            if r["id"] == forced_id:
                default_index = labels.index(
                    f"{r['title']}  —  {r['subject']} • Grade {r['grade']} • {r['type']}"
                ) + 1  # +1 because of placeholder
                break
    except:
        pass

choice = st.selectbox(
    "Choose an item:",
    labels_with_placeholder,
    index=default_index
)

# If still on placeholder → stop
if choice == "— Select an item —":
    st.info("👆 Select a worksheet to begin.")
    st.stop()

item_id = label_map[choice]

# Update URL (so it can be shared)
st.query_params["item"] = str(item_id)

# ----------------------------
# Load item
# ----------------------------
item = get_published_item_by_id(item_id)

st.divider()

_, title, subject, grade, ctype, content, created = item

st.subheader(title)
st.caption(f"{subject} • Grade {grade} • {ctype}")

# ----------------------------
# Render content
# ----------------------------
content_str = (content or "").strip()

if content_str.startswith("{"):
    try:
        data = json.loads(content_str)

        # ✅ NEW: Drive-backed interactive payload
        if (ctype or "").lower() == "interactive" and isinstance(data, dict) and data.get("provider") == "gdrive":
            download_url = data.get("download_url")

            if not download_url:
                st.error("Interactive payload missing download_url.")
                st.json(data)
                st.stop()

            worksheet = load_interactive_from_gdrive(download_url)

            if isinstance(worksheet, dict) and worksheet.get("_error"):
                st.error(f"Could not load worksheet: {worksheet['_error']}")
                st.json(data)
                st.stop()

            if isinstance(worksheet, dict) and worksheet.get("type") == "questions":
                render_interactive_questions(worksheet)
            else:
                st.warning("Downloaded JSON is not a questions worksheet.")
                st.json(worksheet)

        # ✅ Existing direct JSON worksheet
        elif isinstance(data, dict) and data.get("type") == "questions":
            render_interactive_questions(data)

        else:
            st.json(data)

    except Exception:
        st.error("This content looks like JSON but could not be parsed.")
        st.markdown(translate_latex(content), unsafe_allow_html=True)

else:
    st.markdown(translate_latex(content), unsafe_allow_html=True)
