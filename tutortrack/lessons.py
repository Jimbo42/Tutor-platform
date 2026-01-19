from datetime import datetime, date
from pathlib import Path
import sqlite3
import pyperclip
import streamlit as st
import pandas as pd
import re
import json
from openai import Client
from streamlit import session_state as ss

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "AIDA.db"
SHORTCODE_FILE = BASE_DIR / "shortcodes.json"

def get_conn():
    return sqlite3.connect(DB_PATH)

def get_student_list():
    conn_l = get_conn()
    cl = conn_l.cursor()
    with conn_l:
        cl.execute("SELECT Name FROM Students ORDER BY Name ASC")
        data = cl.fetchall()
        ss.name_list = [row[0] for row in data]

    conn_l.close()

def get_student_data():
    conn_b = get_conn()
    cb = conn_b.cursor()
    with conn_b:
        cb.execute("SELECT Course_Goals, Notes, Oxford_Link, Slug FROM Students Where Name = :name", {"name": ss.edit_student})
        ss.student_data = cb.fetchall()

        cb.execute("SELECT * FROM Lessons Where Student = :name ORDER BY Lesson_Date DESC", {"name": ss.edit_student})
        ss.lesson_data = cb.fetchall()

        cb.execute("SELECT Book, Page, ROWID FROM Books Where Student = :name ORDER BY Book", {"name": ss.edit_student})
        books = cb.fetchall()
        # Create a DataFrame with column names
        ss.book_data = pd.DataFrame(books, columns=["Book", "Page", "ROWID"])
        ss.book_data["Complete"] = False
        ss.book_data["Remove"] = False

    conn_b.close()

def update_student_data( goals, notes, link, slug):
    # Update the database
    conn = get_conn()
    c = conn.cursor()

    # SQL query to update the student's record
    with conn:
        c.execute("""
            UPDATE Students
            SET Course_Goals = :course_goals, Notes = :notes, Oxford_Link = :oxford_link, Slug = :slug
            WHERE Name = :student_name
        """, {
            "course_goals": goals,
            "notes": notes,
            "oxford_link": link,
            "slug": slug,
            "student_name": ss.edit_student
        })

        if c.rowcount > 0:
            st.success(f"Details for {ss.edit_student} updated successfully!")
    conn.close()

    # Optionally, refresh the student data after updating
    get_student_data()

def show_lesson_table():
    if not ss.lesson_data:
        st.info("No lessons available.")
        return

    # Build a DataFrame from lesson_data
    df = pd.DataFrame(
        ss.lesson_data,
        columns=[
            "Daily Notes",
            "Date",
            "Goals",
            "Quick Notes",
            "Col4",
            "Col5",
            "Col6",
            "Exercises",
            "RowID",
        ]
    )

    # Keep only what you want to display
    df = df[["Date", "Goals", "Exercises", "Quick Notes"]]

    # Optional: nicer date formatting
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        height=420
    )

def load_shortcodes():
    try:
        with open(SHORTCODE_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_shortcodes(shortcodes):
    with open(SHORTCODE_FILE, "w") as f:
        json.dump(shortcodes, f, indent=4)

def add_q_note():

    # Load shortcodes
    shortcode_map = load_shortcodes()

    # --- 2. Expand the shortcodes in ss.q_input ---
    user_text = ss.q_input.strip()

    for code, phrase in shortcode_map.items():
        # replace whole words only, case-insensitive
        pattern = r'\b' + code + r'\b'
        user_text = re.sub(pattern, phrase, user_text, flags=re.IGNORECASE)

    if ss.lesson_data[ss.lesson_number][3]:
        Q_Notes = ss.lesson_data[ss.lesson_number][3] + "," + user_text
    else:
        Q_Notes = user_text
    ss.q_input = ""

    conn_a = get_conn()
    ca = conn_a.cursor()
    with conn_a:
        ca.execute("UPDATE Lessons SET Quick_Notes = :quickNotes where ROWID = :rowID", {"quickNotes": Q_Notes, "rowID": ss.rowID})

    conn_a.close()
    get_student_data()

def add_lesson():
    dateToday = date.today().strftime('%Y-%m-%d %H:%M')
    conn_a = get_conn()
    ca = conn_a.cursor()
    with conn_a:
        ca.execute("INSERT INTO Lessons (Student, Lesson_Date, Lesson_Goals) VALUES (:name, :lessonDate, :LessonGoals)", {"name": ss.edit_student, "lessonDate": dateToday, "LessonGoals": ss.lesson_goal})

    conn_a.close()
    get_student_data()

def add_book():
    conn_a = get_conn()
    ca = conn_a.cursor()
    with conn_a:
        ca.execute("INSERT INTO Books (Student, Book, Page) VALUES (:name, :book, :pagenum)",
                   {"name": ss.edit_student, "book": ss.new_book, "pagenum": 3})

    conn_a.close()
    get_student_data()

def update_lesson_data( goals, dNote, qNotes, exercises):

    # Update the database
    conn = get_conn()
    c = conn.cursor()

    # SQL query to update the lessons record
    with conn:
        c.execute("""
            UPDATE Lessons
            SET Lesson_Goals = :lesson_goals, Daily_Note = :dailyNotes, Quick_Notes = :quickNotes, Exercises = :exercises
            WHERE ROWID = :rowID
        """, {
            "lesson_goals": goals,
            "dailyNotes": dNote,
            "quickNotes": qNotes,
            "exercises": exercises,
            "rowID": ss.rowID
        })

    if c.rowcount > 0:
        st.success(f"Details for {ss.edit_student} updated successfully!")

    conn.close()

    st.success(f"Lesson for {ss.edit_student} updated successfully!")

    # Optionally, refresh the student data after updating
    get_student_data()

def update_book():
    # Check for edited rows
    _changed = False
    updated_value = None

    if 'edited_rows' in ss.currBook:
        for row_index, changes in ss.currBook['edited_rows'].items():
            # If the Complete checkbox is set to True
            if changes.get('Complete', False):
                # Retrieve the corresponding row from the original data
                row_data = ss.book_data.iloc[row_index]
                rowID = int(row_data['ROWID'])
                currPage = row_data['Page']
                currBook = row_data['Book']
                nextPage = int(currPage + 1)
                exerciseText = currBook + "/" + str(currPage)

                if ss.lesson_data[ss.lesson_number][7]:
                    currentExerciseText = ss.lesson_data[ss.lesson_number][7]
                else:
                    currentExerciseText = ""
                # Split the current field value into a list of Book/Page pairs
                exercises_list = currentExerciseText.split(',')

                # Separate Book and Page from the input text
                input_book, input_page = exerciseText.split('/')

                # Initialize a flag to track whether the input book exists in the exercises list
                book_exists = False

                # Iterate through the existing exercises to check if the input book already exists
                for i, exercise in enumerate(exercises_list):
                    if len(exercise) > 1:
                        book, page = exercise.split('/')
                        if book == input_book:
                            # Split the existing page range if it's a range (e.g., Y62/4-5)
                            page_range = page.split('-')
                            if len(page_range) == 2:
                                start_page, end_page = map(int, page_range)
                                if int(input_page) == end_page + 1:
                                    exercises_list[i] = f"{book}/{start_page}-{input_page}"
                                    book_exists = True
                                    break
                            elif int(page) + 1 == int(input_page):
                                exercises_list[i] = f"{book}/{page}-{input_page}"
                                book_exists = True
                                break

                # If the input book doesn't exist, add it to the exercises list
                if not book_exists:
                    exercises_list.append(exerciseText)

                # Join the updated exercises list into a single string
                updated_value = ','.join(exercises_list)
                _changed = True

            elif changes.get('Page'):
                row_data = ss.book_data.iloc[row_index]
                rowID = int(row_data['ROWID'])
                nextPage = changes.get('Page')
                updated_value = None
                _changed = True

            _success = False
            if _changed:

                conn_a = get_conn()
                try:
                    ca = conn_a.cursor()

                    # Execute the SQL query to update the Page
                    with conn_a:
                        ca.execute(
                            "UPDATE Books SET Page = :nextPage WHERE ROWID = :rowID",
                            {"nextPage": nextPage, "rowID": rowID}
                        )

                    # Check if the update affected any rows
                    if ca.rowcount > 0:
                        _success = True
                    else:
                        st.warning("No rows were updated. Check if the ROWID exists.")

                except sqlite3.Error as e:
                    st.error(f"An error occurred: {e}")

                if _success and updated_value:
                    try:
                        ca = conn_a.cursor()

                        # Execute the SQL query to update the Page
                        with conn_a:
                            ca.execute(
                                "UPDATE Lessons SET Exercises = :exercises WHERE ROWID = :rowID",
                                {"exercises": updated_value, "rowID": ss.rowID}
                            )
                        # Check if the update affected any rows
                        if ca.rowcount > 0:
                            st.success("Update successful!")
                        else:
                            st.warning("No rows were updated. Check if the ROWID exists.")

                        get_student_data()

                    except sqlite3.Error as e:
                        st.error(f"An error occurred: {e}")


def summarize_lesson():

    openAI_api_key = st.secrets["OPENAI_API_KEY"]

    prompt_string = f"""
    You are writing a professional session summary that will be read by parents and the education director.

    Based ONLY on the following tutorial session notes, write a concise, factual summary of the student's progress.

    Rules:
    - Do NOT invent encouragement, praise, or motivational language.
    - ONLY include positive or encouraging remarks if they explicitly appear in the notes.
    - Do NOT use generic praise words like "commendable", "excellent", "great job", etc unless they appear in the notes.
    - Keep the tone professional, natural, and objective.
    - Summarize what was worked on, what concepts were discussed, and how the student progressed.
    - If the notes contain a closing encouragement (e.g. "good luck tomorrow", "nice work today"), you may include it naturally at the end.
    - If the notes do NOT contain encouragement, do not add any.

    Start the summary with:
    "During today's session with {ss.edit_student.split()[0]}, ..."

    Make sure to incorporate the lesson goals naturally:
    - Lesson goals were: {ss.lesson_data[ss.lesson_number][2]}

    Here are the session notes:
    - {ss.lesson_data[ss.lesson_number][3]}
    """

    submitData = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a teaching assistant"},
            {"role": "user", "content": prompt_string}
            ],
        "temperature": 0.0,
        "max_tokens": 480,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    # submit the prompt to the OpenAI API and retrieve the response
    client = Client(api_key=openAI_api_key)
    response_text = ""
    try:
        response = client.chat.completions.create(**submitData)
        for choice in response.choices:
            response_text += choice.message.content.strip()

    except Exception as error:
        print(f"An error occurred: {error}")

    if len(response_text) >0:
        review_note(response_text)

#  Dialog Boxes
@st.dialog("Add Student")
def add_new_student():
    student_name = st.text_input("Student Name")
    if st.button("Submit"):
        dateToday = date.today().strftime('%Y-%m-%d %H:%M')
        conn_a = get_conn()
        ca = conn_a.cursor()
        with conn_a:
            default_link = "https://mis.oxfordlearning.com/class/index/index"
            ca.execute(
                "INSERT INTO Students (Name, Oxford_Link) VALUES (:name, :link)",
                {"name": student_name, "link": default_link})

            ca.execute(
                "INSERT INTO Lessons (Student, Lesson_Date) VALUES (:name, :lessonDate)",
                {"name": student_name, "lessonDate": dateToday})

        conn_a.close()
        st.rerun()


@st.dialog("Remove student")
def remove_student():
    if st.button("Remove all records for:" + ss.edit_student):
        conn_a = get_conn()
        try:
            ca = conn_a.cursor()
            with conn_a:
                ## Delete records from Students table
                ca.execute("DELETE FROM Students WHERE Name = :name", {"name": ss.edit_student})
                if ca.rowcount > 0:
                    ## Delete records from Lessons table
                    ca.execute("DELETE FROM Lessons WHERE Student = :name", {"name": ss.edit_student})

                    ## Delete records from Books table
                    ca.execute("DELETE FROM Books WHERE Student = :name", {"name": ss.edit_student})

        except Exception as e:
            print(f"Error deleting records: {e}")
        finally:
            # remove from Student Selection
            ss.student_list.remove(ss.edit_student)
            ss.edit_student = None
            st.rerun()


@st.dialog("Add lesson")
def add_new_lesson():
    lesson_goal = st.text_input("Lesson Goals")
    if st.button("Submit"):
        ss.lesson_goal = lesson_goal
        add_lesson()
        st.rerun()


@st.dialog("Add Book")
def add_new_book():
    new_book = st.text_input("Book Name")
    if st.button("Add Book"):
        ss.new_book = new_book
        add_book()
        st.rerun()


@st.dialog("Review Note")
def review_note(response_text):

    reviewing_note = st.text_area("Review Note", response_text, height=400)
    if st.button("OK"):
        #copy note to clipboard
        pyperclip.copy(reviewing_note)

        conn_a = get_conn()
        try:
            ca = conn_a.cursor()

            # Execute the SQL query to update the notes
            with conn_a:
                ca.execute(
                    "UPDATE Lessons SET Daily_Note = :notes WHERE ROWID = :rowID",
                    {"notes": reviewing_note, "rowID": ss.rowID}
                )
            # Check if the update affected any rows
            if ca.rowcount > 0:
                st.success("Update successful!")
            else:
                st.warning("No rows were updated. Check if the ROWID exists.")

            get_student_data()
            st.rerun()

        except sqlite3.Error as e:
            st.error(f"An error occurred: {e}")


#  Setting initial parameters for session variables
if 'student_list' not in ss:
    ss.student_list = []
    ss.name_list = None
    ss.edit_student = None
    ss.student_data = None
    ss.lesson_data = None
    ss.book_data = None
    ss.q_input = ""
    ss.edit_details = False

if "lesson_number" not in ss:
    ss.lesson_number = 0
    ss.addLesson = False
    ss.rowID = 0

#  Menu and selection setup
with st.sidebar:
    get_student_list()
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.button("New 🙋", on_click=add_new_student)
    with col_s2:
        if st.button("Leave 🛖"):
            if ss.edit_student:
                ss.student_list.remove(ss.edit_student)
                if ss.student_list:
                    ss.edit_student = ss.student_list[0]
                else:
                    ss.edit_student = None
                st.rerun()

    with st.popover("Student List"):
        st.pills("Student List", ss.name_list, selection_mode="multi", key="pills_selected")

        if st.button("Apply selection"):
            if ss.pills_selected:
                ss.student_list = list(ss.pills_selected)
                ss.edit_student = ss.student_list[0]
                ss.lesson_number = 0
                get_student_data()
                st.rerun()

    if ss.student_list:
        if "student_radio" not in ss:
            ss.student_radio = ss.edit_student

        active_student = st.sidebar.radio("In Class", ss.student_list, key="student_radio")

        if active_student != ss.edit_student:
            ss.edit_student = active_student
            ss.lesson_number = 0
            get_student_data()
            st.rerun()


# Beginning of main script
if ss.edit_student:
    #ss.edit_details = False
    col_A, col_B, col_C = st.columns(3)
    with col_A:
        st.link_button(ss.edit_student, ss.student_data[0][2], type="primary")
    with col_B:
        lesson_date_string = ss.lesson_data[ss.lesson_number][1]
        lesson_date = datetime.strptime(lesson_date_string, "%Y-%m-%d %H:%M").strftime("%B %d, %Y")
        st.text(lesson_date)
    with col_C:
        if ss.student_data[0][3]:
            st.subheader(ss.student_data[0][3], divider="rainbow")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.button("Add 🕜", on_click=add_new_lesson)
    with col2:
        st.button("Write 📝", on_click=summarize_lesson)
    with col3:
        ss.edit_details = not st.button("Edit 🧮")
    with col4:
        st.button("Add 📗", on_click=add_new_book)
    with col5:
        st.button("Remove 🚯", on_click=remove_student)

    num_lessons = len(ss.lesson_data)

    q_input = st.text_input("Quick Add:", key="q_input", on_change=add_q_note)

    with st.expander("🔤 Quick Note Codes"):
        shortcodes = load_shortcodes()

        items_html = ""
        for code, phrase in shortcodes.items():
            items_html += (
                f"<div class='qc-item'><b>{code}</b><br>{phrase}</div>"
            )

        st.markdown(
            f"""
            <style>
            .qc-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 6px 12px;
            }}
            .qc-item {{
                font-size: 11.5px;
                line-height: 1.25;
            }}
            </style>
            <div class="qc-grid">
                {items_html}
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.expander("", expanded=not ss.edit_details, icon="📚"):
        column_config = {
            "Book": "Book Title",  # Title for the 'Book' column
            "Page": st.column_config.NumberColumn(
                "Page Number",  # Title for the 'Page' column
                step=1
            ),
            "ROWID": None,  # Hide the 'ROWID' column
            "Complete": st.column_config.CheckboxColumn(
                "Complete",  # Title for the 'Complete' column
            ),
            "Remove": st.column_config.CheckboxColumn(
                "Remove"  # Title for the 'Remove' column
            )
        }

        st.data_editor(
            ss.book_data,
            column_config=column_config,
            width='stretch',
            hide_index=True,
            on_change=update_book,
            key="currBook"
        )

    with st.form("Lesson Notes"):
        ss.rowID = ss.lesson_data[ss.lesson_number][8]
        lesson_goals = st.text_input("Lesson Goals:", ss.lesson_data[ss.lesson_number][2])
        exercises = st.text_input("Exercises:", ss.lesson_data[ss.lesson_number][7])
        daily_note = st.text_area("Daily Notes:", ss.lesson_data[ss.lesson_number][0])
        quick_notes = st.text_area("Quick Notes:", ss.lesson_data[ss.lesson_number][3])

        if st.form_submit_button("Save Lesson", disabled=ss.edit_details):
            update_lesson_data(lesson_goals, daily_note, quick_notes, exercises)

    with (st.expander("", expanded=not ss.edit_details, icon="🐋")):
        with st.form("Student Details"):
            course_goals = st.text_input("Course Goals", ss.student_data[0][0])
            notes = st.text_area("Notes", ss.student_data[0][1])
            oxford_link = st.text_input("Oxford Link", ss.student_data[0][2])
            slug = st.text_input("Update Required:", ss.student_data[0][3])

            if st.form_submit_button("Save Details", disabled=ss.edit_details):
                update_student_data(course_goals, notes, oxford_link, slug)
    with (st.expander("", expanded=not ss.edit_details, icon="📋")):
        show_lesson_table()

