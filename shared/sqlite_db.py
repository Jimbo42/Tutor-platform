from pathlib import Path
import sqlite3

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "AIDA.db"

def get_conn():
    return sqlite3.connect(DB_PATH)
