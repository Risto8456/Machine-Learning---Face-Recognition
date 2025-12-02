# test_pipeline.py
# 測試腳本
import os
from app.pipeline import FacePipeline
from app.db import init_db, save_embedding, load_all_embeddings
import numpy as np
from numpy.linalg import norm

IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'images')

def cosine_sim(a,b):
    return np.dot(a,b) / (norm(a)*norm(b))

def main():
    init_db()
    pipeline = FacePipeline()
    # 1) 註冊兩張個人照（假設每張圖只有一張臉）
    register_list = [
        ('賴清德', os.path.join(IMAGES_DIR, '賴清德.jpg')),
        ('陳亭妃', os.path.join(IMAGES_DIR, '陳亭妃.jpg')),
    ]
    for name, path in register_list:
        print(f"Processing register: {name} <- {path}")
        results = pipeline.process_image(path)
        if not results:
            print("  no face detected.")
            continue
        # 取第 0 張臉 (若有多張可擴充)
        emb = results[0]['embedding'].astype('float32')
        save_embedding(name, path, emb)
        print(f"  saved embedding for {name}, dim={emb.shape}")

    # 2) load DB
    db = load_all_embeddings()
    print(f"DB contains {len(db)} entries")

    # 3) 對合照做辨識，計算每個偵測臉與 DB 的相似度
    group_path = os.path.join(IMAGES_DIR, '陳亭妃和賴清德合照.webp')
    print(f"Processing group image: {group_path}")
    res = pipeline.process_image(group_path)
    if not res:
        print("No faces detected in group image.")
        return
    for i, item in enumerate(res):
        emb = item['embedding']
        print(f"Face #{i} (prob={item['prob']})")
        # 比對 DB 所有
        sims = []
        for r in db:
            sim = cosine_sim(emb, r['embedding'])
            sims.append((r['name'], sim))
        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
        print("  top matches:")
        for name, s in sims_sorted[:5]:
            print(f"    {name} -> cosine {s:.4f}")

if __name__ == '__main__':
    main()
