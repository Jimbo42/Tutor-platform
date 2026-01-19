import sqlite3
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "published_content.db"

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_published_db():
    conn = get_conn()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS PublishedItems (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        subject TEXT,
        grade TEXT,
        content_type TEXT,
        content TEXT NOT NULL,
        created_at TEXT,
        updated_at TEXT,
        visible INTEGER DEFAULT 1
    );
    """)
    conn.commit()
    conn.close()


# ========== TutorTrack uses this ==========
def publish_item(title, subject, grade, content_type, content):
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_conn()
    conn.execute("""
        INSERT INTO PublishedItems
        (title, subject, grade, content_type, content, created_at, updated_at, visible)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1)
    """, (title, subject, grade, content_type, content, now, now))
    conn.commit()
    conn.close()


# ========== TutorAssist uses these ==========
def get_published_items():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, subject, grade, content_type, created_at
        FROM PublishedItems
        WHERE visible = 1
        ORDER BY created_at DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def get_published_item_by_id(item_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, subject, grade, content_type, content, created_at
        FROM PublishedItems
        WHERE id = ? AND visible = 1
    """, (item_id,))
    row = cur.fetchone()
    conn.close()
    return row

def get_distinct_values():
    conn = get_conn()
    cur = conn.cursor()

    subjects = [r[0] for r in cur.execute("SELECT DISTINCT subject FROM PublishedItems WHERE visible=1 AND subject IS NOT NULL")]
    grades   = [r[0] for r in cur.execute("SELECT DISTINCT grade FROM PublishedItems WHERE visible=1 AND grade IS NOT NULL")]
    types    = [r[0] for r in cur.execute("SELECT DISTINCT content_type FROM PublishedItems WHERE visible=1 AND content_type IS NOT NULL")]

    conn.close()
    return subjects, grades, types

def list_published_items():
    conn = get_conn()
    rows = conn.execute("""
        SELECT id, title, subject, grade, content_type, created_at, updated_at, visible
        FROM PublishedItems
        ORDER BY created_at DESC
    """).fetchall()
    conn.close()
    return rows

def get_published_item_full(item_id):
    conn = get_conn()
    row = conn.execute("""
        SELECT id, title, subject, grade, content_type, content, created_at, updated_at, visible
        FROM PublishedItems WHERE id=?
    """, (item_id,)).fetchone()
    conn.close()
    return row

def update_published_item(item_id, title, subject, grade, content_type, content, visible):
    conn = get_conn()
    conn.execute("""
        UPDATE PublishedItems
        SET title=?, subject=?, grade=?, content_type=?, content=?, updated_at=datetime('now'), visible=?
        WHERE id=?
    """, (title, subject, grade, content_type, content, visible, item_id))
    conn.commit()
    conn.close()

def delete_published_item(item_id):
    conn = get_conn()
    conn.execute("DELETE FROM PublishedItems WHERE id=?", (item_id,))
    conn.commit()
    conn.close()
