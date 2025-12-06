# app/detector.py
# MTCNN 初始化與 detect helper
from facenet_pytorch import MTCNN
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class FaceDetector:
    def __init__(self, device=None, keep_all=True, min_prob=0.95):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_prob = min_prob
        # keep_all: True -> 回傳所有偵測到的臉
        self.mtcnn = MTCNN(keep_all=keep_all, device=self.device)

    def detect(self, pil_image):
        """
        使用 mtcnn.detect 回傳 boxes, probs, landmarks
        pil_image: PIL.Image
        過濾掉低於 min_prob 的臉
        """
        boxes, probs, landmarks = self.mtcnn.detect(pil_image, landmarks=True)
        # boxes: Nx4, landmarks: Nx5x2 (left_eye,right_eye,nose,left_mouth,right_mouth)
        
        # 若沒有偵測到任何臉
        if boxes is None or len(boxes) == 0:
            return None, None, None
        
        # 過濾低於 min_prob 的臉
        keep_indices = [i for i, p in enumerate(probs) if p >= self.min_prob]
        if len(keep_indices) == 0:
            return None, None, None

        boxes_filtered = boxes[keep_indices]
        probs_filtered = probs[keep_indices]
        landmarks_filtered = landmarks[keep_indices]

        return boxes_filtered, probs_filtered, landmarks_filtered


# -----------------------------
# Main 測試程式
# -----------------------------
if __name__ == "__main__":
    
    # 輸入圖片路徑
    image_path = "images/input/議會畫面.png"

    # 讀圖 (支援中文路徑)
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Cannot open image: {e}")
        exit(0)

    # 初始化 detector
    detector = FaceDetector()

    # 偵測
    boxes, probs, landmarks = detector.detect(img)

    # 複製一份用來畫圖
    draw = ImageDraw.Draw(img)

    # 使用內建字體，可調整大小
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except:
        font = ImageFont.load_default()

    if boxes is not None:
        for i, box in enumerate(boxes):
            score = probs[i]

            x1, y1, x2, y2 = box
            # 畫人臉框
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            # 計算文字位置，確保不超出圖片
            text = f"{score*100:.2f}%"
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                text_width, text_height = font.getsize(text)
            
            text_x = x1
            text_y = max(0, y1 - text_height - 2)  # 上方，如果超過就貼到 y=0
            draw.text((text_x, text_y), text, fill="yellow", font=font)
            print(probs[i])

            # 畫 landmarks
            lm = landmarks[i]
            for (x, y) in lm:
                r = 2
                draw.ellipse((x-r, y-r, x+r, y+r), fill="green")

    # 顯示結果
    img.show()