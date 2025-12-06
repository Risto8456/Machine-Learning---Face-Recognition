# recognize_faces.py
# 每次辨識
import os
from app.pipeline import FacePipeline
from app.recognizer import FaceRecognizer
from app.db import load_all_embeddings

INPUT_IMAGE = "images/input/議會畫面.png"        # 要辨識的合照
OUTPUT_DIR = "result"           # 儲存剪下的人臉 + 前 3 名的名字&相似度

def main():
    pipeline = FacePipeline()

    # 1) 載入 DB 所有註冊人
    db = load_all_embeddings()
    print(f"Loaded {len(db)} registered faces.")

    # 2) 對合照做人臉 pipeline
    results = pipeline.process_image(INPUT_IMAGE)
    if not results:
        print("No faces detected.")
        return

    print(f"Detected {len(results)} faces in input image.")

    # 3) 建立比對器
    recognizer = FaceRecognizer(db_embeddings=db, output_dir=OUTPUT_DIR)

    # 4) 做前 3 名比對 + 儲存影像
    full_results = recognizer.recognize_faces(results)

    # 5) 印出結果
    for item in full_results:
        print(f"\nFace #{item['face_id']}")
        print("  Top 3 matches:")
        for rank, (name, score, img_path) in enumerate(item["top3"], 1):
            print(f"    {rank}. {name} {score*100:.2f}% -> saved: {img_path}")

if __name__ == "__main__":
    main()
