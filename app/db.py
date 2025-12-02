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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, name, img_path, embedding FROM faces')
    rows = c.fetchall()
    res = []
    for r in rows:
        emb = np.frombuffer(r[3], dtype=np.float32)
        res.append({'id': r[0], 'name': r[1], 'img_path': r[2], 'embedding': emb})
    conn.close()
    return res
