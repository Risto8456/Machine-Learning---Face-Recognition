# app/pipeline.py
# 把整段流程串起來
from .detector import FaceDetector
from .aligner import FaceAligner
from .embedder import FaceEmbedder
from PIL import Image
import numpy as np

class FacePipeline:
    def __init__(self, device=None):
        self.detector = FaceDetector(device=device, keep_all=True)
        self.aligner = FaceAligner(expand_ratio=1.4)
        self.embedder = FaceEmbedder(device=device)

    def process_image(self, img_path, re_detect_after_rotate=True):
        """
        輸入：image path
        回傳：list of dict 每個 dict 包含 {box, prob, landmarks, aligned_face(PIL), embedding(np.array)}
        流程：
         1. 讀圖 PIL
         2. mtcnn.detect (boxes, probs, landmarks)
         3. 對每張臉：aligner.align_face (旋轉+裁剪)，然後（選項）在 rotated 圖上重新 run mtcnn 以取得更精確 crop (若 re_detect_after_rotate True)
         4. embedder.embed
        """
        pil = Image.open(img_path).convert('RGB')
        boxes, probs, landmarks = self.detector.detect(pil)
        results = []
        if boxes is None:
            return results

        for i, box in enumerate(boxes):
            lm = landmarks[i]
            prob = probs[i]
            # 旋轉+裁剪
            _, aligned = self.aligner.align_face(pil, box, lm)

            if re_detect_after_rotate:
                new_boxes, new_probs, new_landmarks = self.detector.detect(aligned)
                if new_boxes is not None and len(new_boxes) > 0:
                    # 取概率最大的一張
                    max_idx = np.argmax(new_probs)
                    new_box = new_boxes[max_idx]
                    aligned = aligned.crop(new_box)

            emb = self.embedder.embed(aligned)
            results.append({
                'box': box,
                'prob': float(prob) if prob is not None else None,
                'landmarks': lm,
                'aligned': aligned,
                'embedding': emb
            })
        return results
