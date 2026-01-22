import sqlite3
import pandas as pd
import streamlit as st
from streamlit import session_state as ss
import json
from pathlib import Path
from datetime import datetime

# ==============================
# ChatTemplates Admin Functions
# ==============================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "AIDA.db"
# short codes for lesson notes
SHORTCODE_FILE = BASE_DIR / "shortcodes.json"

def get_conn():
    return sqlite3.connect(DB_PATH)

def get_chattemplates_df():
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT id, title, category, model, updated_at
        FROM ChatTemplates
        ORDER BY title
        """,
        conn
    )
    conn.close()

    df.insert(0, "Edit", False)
    df.insert(1, "Delete", False)
    df.insert(2, "Export", False)

    return df

def get_chattemplate_by_id(tpl_id: int):
    conn = get_conn()
    cur = conn.cursor()
    row = cur.execute("""
        SELECT id, title, category, model, system_prompt, user_prompt, fields_json, params_json
        FROM ChatTemplates WHERE id=?
    """, (tpl_id,)).fetchone()
    conn.close()

    if not row:
        return None

    return {
        "id": row[0],
        "title": row[1],
        "category": row[2],
        "model": row[3],
        "system_prompt": row[4],
        "user_prompt": row[5],
        "fields": json.loads(row[6] or "[]"),
        "params": json.loads(row[7] or "{}"),
    }

def upsert_chattemplate(tpl: dict):
    required = ["title", "model", "system_prompt", "user_prompt", "fields", "params"]
    for k in required:
        if k not in tpl:
            raise ValueError(f"Missing required key: {k}")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_conn()
    cur = conn.cursor()

    if tpl.get("id"):
        # -----------------------------
        # UPDATE existing template
        # -----------------------------
        cur.execute("""
            UPDATE ChatTemplates
            SET
                title = ?,
                category = ?,
                model = ?,
                system_prompt = ?,
                user_prompt = ?,
                fields_json = ?,
                params_json = ?,
                updated_at = ?
            WHERE id = ?
        """, (
            tpl["title"].strip(),
            tpl.get("category"),
            tpl["model"].strip(),
            tpl["system_prompt"],
            tpl["user_prompt"],
            json.dumps(tpl.get("fields") or [], ensure_ascii=False),
            json.dumps(tpl.get("params") or {}, ensure_ascii=False),
            now,
            tpl["id"],
        ))

    else:
        # -----------------------------
        # INSERT new template
        # -----------------------------
        cur.execute("""
            INSERT INTO ChatTemplates
            (title, category, model, system_prompt, user_prompt, fields_json, params_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tpl["title"].strip(),
            tpl.get("category"),
            tpl["model"].strip(),
            tpl["system_prompt"],
            tpl["user_prompt"],
            json.dumps(tpl["fields"], ensure_ascii=False),
            json.dumps(tpl["params"], ensure_ascii=False),
            now
        ))

    conn.commit()
    conn.close()

def delete_chattemplate_by_id(tpl_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM ChatTemplates WHERE id=?", (tpl_id,))
    conn.commit()
    conn.close()

def export_chattemplate_to_json(tpl_id: int) -> str:
    tpl = get_chattemplate_by_id(tpl_id)
    if not tpl:
        raise ValueError("Template not found")

    portable = {
        "title": tpl["title"],
        "category": tpl.get("category"),
        "model": tpl["model"],
        "system_prompt": tpl["system_prompt"],
        "user_prompt": tpl["user_prompt"],
        "fields": tpl.get("fields") or [],
        "params": tpl.get("params") or {},
    }

    return json.dumps(portable, ensure_ascii=False, indent=2)

def import_chattemplate_from_json_file(uploaded_file):
    tpl = json.loads(uploaded_file.read().decode("utf-8"))

    # ---- Normalize & validate ----
    required = ["title", "model", "system_prompt", "user_prompt"]
    for k in required:
        if k not in tpl:
            raise ValueError(f"Template JSON missing required key: {k}")

    tpl_obj = {
        "title": tpl["title"],
        "category": tpl.get("category"),
        "model": tpl["model"],
        "system_prompt": tpl["system_prompt"],
        "user_prompt": tpl["user_prompt"],
        "fields": tpl.get("fields") or [],
        "params": tpl.get("params") or {},
    }

    # Validate JSON blocks
    if not isinstance(tpl_obj["fields"], list):
        raise ValueError("fields must be a JSON array")

    if not isinstance(tpl_obj["params"], dict):
        raise ValueError("params must be a JSON object")

    upsert_chattemplate(tpl_obj)

def import_chattemplates_from_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)

    required = {"title", "category", "model", "system_prompt", "user_prompt", "fields_json", "params_json"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    count = 0
    for _, row in df.iterrows():
        tpl = {
            "title": str(row["title"]).strip(),
            "category": row.get("category"),
            "model": str(row["model"]).strip(),
            "system_prompt": row["system_prompt"],
            "user_prompt": row["user_prompt"],
            "fields": json.loads(row["fields_json"] or "[]"),
            "params": json.loads(row["params_json"] or "{}"),
        }
        upsert_chattemplate(tpl)
        count += 1

    return count

@st.dialog("✏️ Edit ChatTemplate")
def edit_chattemplate_dialog(tpl: dict, mode: str = "edit"):
    """
    tpl: template dict from get_chattemplate_by_id() or empty dict for new
    mode: "edit" or "new"
    """

    st.markdown("### ChatTemplate Editor")

    # Defaults
    title = st.text_input("Title", value=tpl.get("title", ""))
    category = st.text_input("Category", value=tpl.get("category", "") or "")
    model = st.text_input("Model", value=tpl.get("model", "gpt-4o-mini"))

    st.divider()

    st.markdown("#### System Prompt")
    system_prompt = st.text_area(
        "System Prompt",
        value=tpl.get("system_prompt", ""),
        height=200
    )

    st.markdown("#### User Prompt")
    user_prompt = st.text_area(
        "User Prompt",
        value=tpl.get("user_prompt", ""),
        height=200
    )

    st.divider()

    st.markdown("#### Fields JSON (controls the UI inputs)")

    fields_json_str = st.text_area(
        "Fields JSON",
        value=json.dumps(tpl.get("fields", []), indent=2, ensure_ascii=False),
        height=200
    )

    st.markdown("#### Params JSON (model overrides)")

    params_json_str = st.text_area(
        "Params JSON",
        value=json.dumps(tpl.get("params", {}), indent=2, ensure_ascii=False),
        height=150
    )

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        if st.button("💾 Save Template", width="stretch"):
            if not title.strip():
                st.error("Title is required.")
                return

            # Validate JSON blocks
            try:
                fields = json.loads(fields_json_str or "[]")
            except Exception as e:
                st.error(f"Fields JSON is invalid: {e}")
                return

            try:
                params = json.loads(params_json_str or "{}")
            except Exception as e:
                st.error(f"Params JSON is invalid: {e}")
                return

            tpl_obj = {
                "id": tpl.get("id"),
                "title": title.strip(),
                "category": category.strip() or None,
                "model": model.strip(),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "fields": fields,
                "params": params,
            }

            upsert_chattemplate(tpl_obj)

            st.success("Template saved.")
            st.rerun()

    with c2:
        if st.button("Cancel", width="stretch"):
            st.rerun()

@st.dialog("🗑️ Confirm Delete")
def confirm_delete_template(tpl_id: int, title: str):
    st.warning(f"Are you sure you want to delete:\n\n**{title}**")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("❌ Cancel"):
            st.rerun()

    with c2:
        if st.button("🗑️ Yes, Delete"):
            delete_chattemplate_by_id(tpl_id)
            st.toast("Template deleted", icon="🗑️")
            st.rerun()

def load_shortcodes():
    try:
        with open(SHORTCODE_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_shortcodes(shortcodes):
    with open(SHORTCODE_FILE, "w") as f:
        json.dump(shortcodes, f, indent=4)

# Function to fetch all lessons from the database
def get_all_lessons():
    conn = get_conn()
    cursor = conn.cursor()
    with conn:
        cursor.execute("""
            SELECT Student, Lesson_Date, Lesson_Goals, Quick_Notes, Daily_Note, ROWID 
            FROM Lessons 
            ORDER BY Student, Lesson_Date DESC
        """)
        lessons = cursor.fetchall()
    conn.close()

    # Convert to a DataFrame
    columns = ["Student", "Lesson_Date", "Lesson_Goals", "Quick_Notes", "Daily_Note", "ROWID"]
    df = pd.DataFrame(lessons, columns=columns)

    # Add a "Delete" checkbox column, default False
    df.insert(0, "Delete", False)

    return df


# Function to update lessons in the database
def update_lessons(updated_df):
    conn = get_conn()
    cursor = conn.cursor()
    with conn:
        for _, row in updated_df.iterrows():
            cursor.execute("""
                UPDATE Lessons 
                SET Student = ?, Lesson_Goals = ?, Quick_Notes = ?, Daily_Note = ?
                WHERE ROWID = ?
            """, (row["Student"], row["Lesson_Goals"], row["Quick_Notes"], row["Daily_Note"], row["ROWID"]))
    conn.close()

# Function to delete selected lessons from the database
def delete_selected_lessons(selected_row_ids):
    if not selected_row_ids:
        return

    conn = get_conn()
    cursor = conn.cursor()
    with conn:
        cursor.executemany("DELETE FROM Lessons WHERE ROWID = ?", [(row_id,) for row_id in selected_row_ids])
    conn.close()

REQUIRED_FIELDS = {"category", "title", "latex", "description"}
OPTIONAL_FIELDS = {"units"}

def validate_formula_csv(df: pd.DataFrame):

    missing = REQUIRED_FIELDS - set(df.columns)
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"

    # If units column exists, validate JSON
    if "units" in df.columns:
        for i, val in enumerate(df["units"]):
            if pd.isna(val) or str(val).strip() == "":
                continue
            try:
                obj = json.loads(val)
                if not isinstance(obj, dict):
                    return False, f"Row {i+1}: units must be a JSON object"
            except Exception as e:
                return False, f"Row {i+1}: invalid units JSON: {e}"

    return True, None

def insert_formulas_from_df(df):
    conn = get_conn()
    cur = conn.cursor()

    has_units = "units" in df.columns

    for _, row in df.iterrows():
        units_val = "{}"
        if has_units and not pd.isna(row["units"]) and str(row["units"]).strip():
            units_val = row["units"]

        cur.execute(
            """
            INSERT INTO Formulas (category, title, latex, description, units)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(row["category"]).strip(),
                str(row["title"]).strip(),
                str(row["latex"]).strip(),
                str(row["description"]).strip(),
                units_val
            )
        )

    conn.commit()
    conn.close()

def formula_csv_importer():
    st.subheader("📥 Import Formulas from CSV")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        accept_multiple_files=False,
        key="formula_csv_uploader"
    )

    if uploaded_file is None:
        st.info("Upload a CSV file with columns: category, title, latex, description, (optional) units")
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV file: {e}")
        return

    valid, error = validate_formula_csv(df)
    if not valid:
        st.error(error)
        return

    st.success(f"CSV valid — {len(df)} formulas detected")

    with st.expander("Preview data"):
        st.dataframe(df, width="stretch")

    if st.button("✅ Import into Formulas table"):
        insert_formulas_from_df(df)

        # ✅ Success feedback
        st.toast("Formulas successfully imported", icon="✅")

        # 🔥 Clear uploader so it doesn't re-trigger
        if "formula_csv_uploader" in ss:
            del ss["formula_csv_uploader"]

        st.rerun()

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
    st.markdown("⚙️ Configuration")
with col2:
    title = st.empty()

if "configMode" not in ss:
    ss.configMode = None
    ss.templates = None

with st.sidebar:
    if st.button("Templates 🍸"):
        ss.configMode = "Templates"
        title.markdown("🧠 ChatTemplates Manager")
    if st.button("Shortcodes "):
        ss.configMode = "Shortcodes"
        title.markdown("✂️ Shortcodes")
    if st.button("All lessons"):
        ss.configMode= "All lessons"
        title.markdown("📋 All Lessons")
    if st.button("Import formulas"):
        ss.configMode= "Formula Import"
        title.markdown("📥 Formula Import")

if ss.configMode == "Templates":

    df = get_chattemplates_df()

    if st.button("➕ New Template"):
        empty = {
            "title": "",
            "category": "",
            "model": "gpt-4o-mini",
            "system_prompt": "",
            "user_prompt": "",
            "fields": [],
            "params": {}
        }
        edit_chattemplate_dialog(empty, mode="new")

    edited = st.data_editor(
        df,
        hide_index=True,
        height=500,
        width="stretch",
        column_config={
            "Edit": st.column_config.CheckboxColumn("✏️ Edit"),
            "Delete": st.column_config.CheckboxColumn("🗑️ Delete"),
            "Export": st.column_config.CheckboxColumn("📤 Export"),
        }
    )

    # Handle actions
    for i, row in edited.iterrows():
        tpl_id = int(row["id"])

        if row["Edit"]:
            tpl = get_chattemplate_by_id(tpl_id)
            edit_chattemplate_dialog(tpl, mode="edit")

        if row["Delete"]:
            confirm_delete_template(tpl_id, row["title"])

        if row["Export"]:
            data = export_chattemplate_to_json(tpl_id)
            st.download_button(
                "Download JSON",
                data=data,
                file_name=f"{row['title'].replace(' ', '_')}.json",
                mime="application/json"
            )

    st.divider()

    st.subheader("📥 Import Templates")

    col1, col2 = st.columns(2)

    with col1:
        up_json = st.file_uploader("Import from JSON", type=["json"])
        if up_json:
            import_chattemplate_from_json_file(up_json)
            st.success("Imported template.")
            st.rerun()

    with col2:
        up_csv = st.file_uploader("Import from CSV", type=["csv"])
        if up_csv:
            n = import_chattemplates_from_csv(up_csv)
            st.success(f"Imported {n} templates.")
            st.rerun()

elif ss.configMode == "Shortcodes":
    shortcodes = load_shortcodes()

    # --- Display existing shortcodes ---
    st.subheader("Existing Shortcodes")
    if shortcodes:
        for code, phrase in shortcodes.items():
            cols = st.columns([2, 6, 1, 1])
            cols[0].write(f"**{code}**")
            cols[1].write(phrase)

            # Edit button
            if cols[2].button("✏️ Edit", key=f"c_edit_{code}"):
                ss.editing_code = code

            # Delete button
            if cols[3].button("🗑️ Delete", key=f"delete_{code}"):
                del shortcodes[code]
                save_shortcodes(shortcodes)
                st.rerun()
    else:
        st.info("No shortcodes defined yet.")

    # --- Editing Form ---
    if "editing_code" in ss:
        code_to_edit = ss.editing_code
        st.subheader(f"Edit Shortcode: {code_to_edit}")

        new_phrase = st.text_input("Phrase", value=shortcodes[code_to_edit])
        if st.button("Save Changes"):
            shortcodes[code_to_edit] = new_phrase
            save_shortcodes(shortcodes)
            del ss.editing_code
            st.rerun()

        if st.button("Cancel"):
            del ss.editing_code
            st.rerun()

    # --- Add New Shortcode ---
    st.subheader("Add New Shortcode")

    new_code = st.text_input("Shortcode (e.g., 'wits')", key="new_shortcode")
    new_phrase = st.text_input("Full Phrase", key="new_phrase")

    if st.button("Add Shortcode"):
        if new_code.strip() == "" or new_phrase.strip() == "":
            st.error("Both fields are required.")
        elif new_code in shortcodes:
            st.error(f"Shortcode '{new_code}' already exists.")
        else:
            shortcodes[new_code] = new_phrase
            save_shortcodes(shortcodes)
            st.success(f"Added shortcode '{new_code}'")
            st.rerun()

elif ss.configMode == "All lessons":

    all_lessons = get_all_lessons()
    if all_lessons.empty:
        st.warning("No lessons found in the database.")
    else:
        # Enable inline editing
        edited_lessons = st.data_editor(
            all_lessons,
            num_rows="fixed",  # Prevent adding new rows
            column_config={"ROWID": None},  # Hide ROWID column
            height=800,
            width="stretch"
        )

        # Find rows where "Delete" checkbox is checked
        rows_to_delete = edited_lessons[edited_lessons["Delete"]].copy()

        if st.button("Save Changes"):
            # Delete selected rows first
            if not rows_to_delete.empty:
                delete_selected_lessons(rows_to_delete["ROWID"].tolist())
                st.warning(f"Deleted {len(rows_to_delete)} row(s).")

            # Remove deleted rows from the dataset
            remaining_lessons = edited_lessons[~edited_lessons["Delete"]].copy()

            # Update the database with remaining (edited) rows
            update_lessons(remaining_lessons)
            st.success("Changes saved successfully!")
            st.rerun()  # Refresh table

elif ss.configMode == "Formula Import":
    formula_csv_importer()