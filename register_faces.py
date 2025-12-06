# register_faces.py
# 事先將 register 資料夾全部存入 DB
import os
import sqlite3
import cv2
import numpy as np
from PIL import Image

from app.pipeline import FacePipeline

DB_PATH = "face_db.sqlite"
REGISTER_DIR = "images/register/"


# ----------------------------------------------------
# 初始化 SQLite
# ----------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            name TEXT PRIMARY KEY,
            embedding BLOB
        )
    """)
    conn.commit()
    return conn


# ----------------------------------------------------
# 從資料庫讀取所有人名
# ----------------------------------------------------
def load_db_names(conn):
    c = conn.cursor()
    c.execute("SELECT name FROM faces")
    rows = c.fetchall()
    return set([r[0] for r in rows])


# ----------------------------------------------------
# 刪除 DB 中，但目錄不存在的資料
# ----------------------------------------------------
def clean_removed_faces(conn, current_names):
    db_names = load_db_names(conn)
    removed = db_names - current_names
    if removed:
        print("移除 DB 中，但資料夾已不存在：", removed)

    c = conn.cursor()
    for name in removed:
        c.execute("DELETE FROM faces WHERE name=?", (name,))
    conn.commit()

# ----------------------------------------------------
# 註冊單張人臉 (改用 FacePipeline)
# ----------------------------------------------------
def register_one_face(conn, name, filepath, pipeline: FacePipeline):
    print(f"註冊：{name}")

    # 1️ pipeline 處理圖片
    results = pipeline.process_image(filepath, re_detect_after_rotate=True)

    if len(results) == 0:
        print(f"{name} 未偵測到人臉")
        return

    # 2️ 取 probs 最大的那一張臉
    probs = [r['prob'] for r in results]
    idx = np.argmax(probs)
    aligned = results[idx]['aligned']
    embedding = results[idx]['embedding']

    # 3️ 轉成 bytes 存 DB
    embedding_bytes = embedding.astype("float32").tobytes()

    # 4️ 寫入 DB
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO faces (name, embedding) VALUES (?, ?)",
        (name, embedding_bytes)
    )
    conn.commit()

    print(f"註冊完成：{name}")

# ----------------------------------------------------
# Main 主程式
# ----------------------------------------------------
def main():
    # 初始化 DB
    conn = init_db()
    
    # 初始化 pipeline（包含 detector, aligner, embedder）
    pipeline = FacePipeline()

    # 已存在 DB 的人名
    db_names = load_db_names(conn)

    # 目前資料夾的人名（檔名不含副檔名）
    file_names = []
    for f in os.listdir(REGISTER_DIR):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            file_names.append(os.path.splitext(f)[0])

    folder_names = set(file_names)

    print("目前資料夾人名：", folder_names)
    print("DB 內人名：", db_names)

    # 清理：若 DB 有但資料夾沒有 → 移除
    clean_removed_faces(conn, folder_names)

    # 新增註冊：資料夾有但 DB 沒有 → 新增
    new_faces = folder_names - db_names
    print("需要新增的：", new_faces)

    for name in new_faces:
        filepath = os.path.join(REGISTER_DIR, name + ".jpg")
        if not os.path.exists(filepath):
            # 有可能是 png/jpg/webp
            for ext in [".png", ".jpeg", ".webp"]:
                alt = os.path.join(REGISTER_DIR, name + ext)
                if os.path.exists(alt):
                    filepath = alt
                    break

        register_one_face(conn, name, filepath, pipeline)

    print("\n註冊流程完成！")

    conn.close()


if __name__ == "__main__":
    main()
