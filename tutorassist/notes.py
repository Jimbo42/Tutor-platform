import streamlit as st
import json
from pathlib import Path
from shared.published_db import get_published_items, get_published_item_by_id

BASE_DIR = Path(__file__).resolve().parent
PUBLISHED_PDF_DIR = BASE_DIR.parent / "shared" / "published_files"

st.title("📘 Notes & Documents")

rows = get_published_items()

# Filter only PDFs
pdf_rows = [r for r in rows if r[4] == "pdf"]  # (id, title, subject, grade, type, created)

if not pdf_rows:
    st.info("No published documents yet.")
    st.stop()

labels = [f"{r[1]}" for r in pdf_rows]
id_map = {f"{r[1]}": r[0] for r in pdf_rows}

choice = st.selectbox("Choose a document:", labels,
                      index=None, placeholder="Choose a Document" )

if choice:
    item_id = id_map[choice]

    item = get_published_item_by_id(item_id)
    content = json.loads(item[5])

    filename = content["filename"]
    pdf_path = PUBLISHED_PDF_DIR / filename

    if not pdf_path.exists():
        st.error("File not found on server.")
    else:
        st.pdf(str(pdf_path))
