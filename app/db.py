# app/db.py
# 簡單 SQLite 存取 embeddings
import sqlite3
import numpy as np
import os
import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'face_db.sqlite')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        img_path TEXT,
        embedding BLOB,
        created_at TEXT
    )
    ''')
    conn.commit()
    conn.close()

def save_embedding(name, img_path, embedding):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    emb_blob = embedding.tobytes()
    created = datetime.datetime.utcnow().isoformat()
    c.execute('INSERT INTO faces (name, img_path, embedding, created_at) VALUES (?, ?, ?, ?)',
              (name, img_path, emb_blob, created))
    conn.commit()
    conn.close()

def load_all_embeddings():
    conn = sqlite3.connect("face_db.sqlite")
    c = conn.cursor()
    c.execute("SELECT name, embedding FROM faces")
    rows = c.fetchall()
    result = []
    for name, emb_blob in rows:
        embedding = np.frombuffer(emb_blob, dtype=np.float32)
        # 自動拼出註冊圖路徑
        img_path = os.path.join("images/register", name + ".jpg")  
        result.append({
            "name": name,
            "embedding": embedding,
            "img_path": img_path
        })
    conn.close()
    return result

