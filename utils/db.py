import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "health_bot.db")

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS user_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    username TEXT,
    full_name TEXT,
    symptoms TEXT,
    response TEXT,
    timestamp TEXT
)
""")

conn.commit()


# ----------------------------
# Save function (FIXED)
# ----------------------------
def save_user_query(user_id, username, full_name, symptoms, response):

    symptoms_str = ", ".join(symptoms) if isinstance(symptoms, list) else str(symptoms)

    cursor.execute("""
        INSERT INTO user_history (
            user_id, username, full_name, symptoms, response, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        str(user_id),
        str(username),
        str(full_name),
        symptoms_str,
        response,
        str(datetime.now())
    ))

    conn.commit()

    print("💾 SAVED:", full_name, symptoms_str)