import streamlit as st
from streamlit import session_state as ss
import pyperclip
from pathlib import Path
import base64
import json
from PIL import Image
import io
import pandas as pd
from openai import OpenAI

openAI_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openAI_api_key)

MODEL = "gpt-4.1"
PROMPT = """

Extract lesson information from the handwritten notes.

RULES:
1. Text written under the "Notes:" section must ALWAYS be extracted and added as items in the Notes list.
   - Split the handwritten notes into meaningful bullet-style items.

2. Extract ONLY text that is visually highlighted in yellow.
   - If the highlight is not clearly yellow, do NOT include it.

3. Avoid duplicates if the same text is both handwritten in the Notes section AND highlighted.

4. DO NOT extract text that is boxed, circled, bracketed, underlined, or visually separated unless it is physically inside a yellow highlight.
   - Rectangles, squares, circles, ovals, arrows, borders, scribbles, or brackets do NOT count as highlighting.

5. Return ONLY valid JSON matching this exact format:

{
  "Student": "",
  "LessonGoals": "",
  "Notes": [],
  "Date": "YYYY-MM-DD"
}

Never wrap the JSON in markdown fences (no ```).
"""

# -------------------------------------------------
# PROCESS A SINGLE PNG IMAGE
# -------------------------------------------------
def extract_fields_from_png(png_bytes: bytes) -> dict:
    # Ensure it's a PNG
    try:
        img = Image.open(io.BytesIO(png_bytes))
        if img.format != "PNG":
            raise ValueError("Uploaded file is NOT a PNG")
    except Exception:
        raise ValueError("Uploaded file is NOT a PNG")

    img_b64 = base64.b64encode(png_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    }
                ]
            }
        ]
    )

    raw = response.choices[0].message.content

    # Sometimes comes back chunked
    if isinstance(raw, list):
        raw = "".join(
            block.get("text", "")
            for block in raw
            if isinstance(block, dict) and "text" in block
        )

    cleaned = raw.strip()

    # Remove unwanted ```json fences if present
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON from model:\n\n{cleaned}")


# -------------------------------------------------
# PROCESS ALL PNG FILES IN THE FOLDER
# -------------------------------------------------
def process_folder(folder_path: str | Path):
    results = []

    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    for img_path in folder.iterdir():
        if img_path.suffix.lower() != ".png":
            continue

        try:
            png_bytes = img_path.read_bytes()
            extracted = extract_fields_from_png(png_bytes)
            extracted["SourceFile"] = img_path.name
            results.append(extracted)

        except Exception as e:
            results.append({
                "Student": "",
                "LessonGoals": "",
                "Notes": [f"ERROR processing {img_path.name}: {e}"],
                "Date": "",
                "SourceFile": img_path.name
            })

    return results

def get_note_text():
    """Triggered when a row is selected in the main table."""
    if ss.note_row.selection.rows:
        rowNum = ss.note_row.selection.rows[0]

        notes_text = ss.lesson_table.iloc[rowNum]["Notes"]

        # Split notes into lines
        lines = [n.strip() for n in notes_text.split(",") if n.strip()]

        # Store in session state so the editor table can show them
        ss.note_lines_df = pd.DataFrame({"Line": lines})
        #ss.note_lines_df.insert(0, "Delete", False)
        ss.selected_note_row = rowNum


# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.title("Lesson Note Extractor")

# 🔧 Change this to wherever you want, or make it configurable later
FOLDER = Path("G:/My Drive/Oxford/Lesson Notes")

if "lesson_table" not in ss:
    ss.lesson_table = pd.DataFrame(columns=["Student", "LessonGoals", "Notes", "Date", "Filename"])

# -------------------------------------------------
# Process all PNGs
# -------------------------------------------------
if st.button("📥 Process All Lesson Note Documents"):

    if not FOLDER.exists():
        st.error(f"Folder does not exist: {FOLDER}")
    else:
        files = list(FOLDER.glob("*.png"))

        if not files:
            st.warning("No PNG files found in the folder.")
        else:
            for img_path in files:
                try:
                    png_bytes = img_path.read_bytes()

                    extracted = extract_fields_from_png(png_bytes)

                    ss.lesson_table.loc[len(ss.lesson_table)] = {
                        "Student": extracted.get("Student", ""),
                        "LessonGoals": extracted.get("LessonGoals", ""),
                        "Notes": ", ".join(extracted.get("Notes", [])),
                        "Date": extracted.get("Date", ""),
                        "Filename": img_path.name,
                    }

                except Exception as e:
                    st.error(f"Error processing {img_path.name}: {e}")

            st.success("All documents processed!")

# ------------------------------------------------------------
# Cleanup Button
# ------------------------------------------------------------
if st.button("🗑️ Delete All PNG Files in Folder"):

    if not FOLDER.exists():
        st.error(f"Folder does not exist: {FOLDER}")
    else:
        files = list(FOLDER.glob("*.png"))

        if not files:
            st.info("No PNG files to delete.")
        else:
            for img_path in files:
                try:
                    img_path.unlink()
                except Exception as e:
                    st.error(f"Could not delete {img_path.name}: {e}")

            st.success(f"Deleted {len(files)} PNG files.")

# ------------------------------------------------------------
# Display Table
# ------------------------------------------------------------
st.subheader("📋 Extracted Lesson Notes Table")

if len(ss.lesson_table) > 0:
    edited = st.dataframe(
        ss.lesson_table,
        selection_mode="single-row",
        on_select=get_note_text,
        key="note_row",
        hide_index=True
    )
else:
    st.info("No records yet. Process documents to populate the table.")

# ------------------------------------------------------------
# Edit Notes
# ------------------------------------------------------------
st.subheader("📝 Edit Notes (line by line)")

if "note_lines_df" in ss:

    st.subheader("📝 Edit Notes (select rows to delete)")

    del_lines = st.dataframe(
        ss.note_lines_df,
        selection_mode="multi-row",
        key="lines",
        on_select="rerun"
    )

    st.write("---")

    if st.button("💾 Save and Copy (Remove Selected Lines)"):
        selected_rows = del_lines.selection.rows

        remaining_df = ss.note_lines_df.drop(
            index=selected_rows,
            errors="ignore"
        ).reset_index(drop=True)

        combined = ", ".join(remaining_df["Line"].tolist())

        pyperclip.copy(combined)

        st.success("Saved and copied remaining notes to clipboard!")
