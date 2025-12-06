# app/recognizer.py
# 儲存：剪下的人臉 + 前三相似度
import os
import numpy as np
from numpy.linalg import norm
from PIL import Image

def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-7)

class FaceRecognizer:
    def __init__(self, db_embeddings, output_dir):
        self.db = db_embeddings
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def recognize_faces(self, detected_faces):
        results = []
        for i, face in enumerate(detected_faces):
            emb = face["embedding"]

            # 計算與 DB 所有 embedding 的相似度
            sims = []
            for r in self.db:
                score = cosine_sim(emb, r['embedding'])
                sims.append((r["name"], float(score), r["img_path"]))

            sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
            top3 = sims_sorted[:3]

            # ------------- 儲存剪下的人臉 ----------------
            person_img_path = os.path.join(self.output_dir, f"person{i+1}_img.png")
            face["aligned"].save(person_img_path)

            # ------------- 儲存 scores.txt ----------------
            score_path = os.path.join(self.output_dir, f"person{i+1}_scores.txt")
            with open(score_path, "w", encoding="utf-8") as f:
                for rank, (name, score, _) in enumerate(top3, 1):
                    f.write(f"{rank}. {name} {score*100:.2f}%\n")

            results.append({
                "face_id": i,
                "img_path": person_img_path,
                "score_path": score_path,
                "top3": top3
            })

        return results
