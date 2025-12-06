# app/aligner.py
# 用左右眼計算旋轉、旋轉並裁切：
# 先旋轉全圖、再裁切放大區域。
# 為了更精準，pipeline 稍後會在旋轉後對 rotated 圖片再次 run MTCNN
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math

def rotate_image(img, angle, center=None):
    """旋轉整張圖片，中心可指定，expand=True"""
    return img.rotate(angle, resample=Image.BICUBIC, center=center, expand=True)

class FaceAligner:
    def __init__(self, expand_ratio=1.4):
        """
        expand_ratio: 在原 bounding box 的基礎上擴大幾倍去包含更多頭部
        """
        self.expand_ratio = expand_ratio

    def _get_rotate_angle(self, left_eye, right_eye):
        """
        計算旋轉角度（度數）
        正值 -> 逆時針旋轉
        """
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = math.degrees(math.atan2(dy, dx))
        return -angle  # 旋轉負角度使眼睛水平
    
    def align_face(self, pil_image, box, landmarks):
        """
        pil_image: PIL.Image
        box: [x1, y1, x2, y2] bounding box
        landmarks: 5x2 array (left_eye, right_eye, nose, left_mouth, right_mouth)
        return: angle
        return: cropped_aligned_face (PIL.Image)
        """

        left_eye = landmarks[0]
        right_eye = landmarks[1]

        # 計算旋轉角度
        angle = self._get_rotate_angle(left_eye, right_eye)

        # 計算原 bounding box 中心
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # 旋轉整張圖片
        rotated = rotate_image(pil_image, -angle, center=(cx, cy))

        # Pillow rotate(expand=True) 後，原圖片會被擴展，左上角偏移
        offset_x = (rotated.width - pil_image.width) / 2
        offset_y = (rotated.height - pil_image.height) / 2

        # 旋轉後方框中心
        cx_rot = cx + offset_x
        cy_rot = cy + offset_y

        # 計算裁切區域 (expand_ratio)
        new_w = w * self.expand_ratio
        new_h = h * self.expand_ratio
        new_x1 = int(cx_rot - new_w / 2)
        new_y1 = int(cy_rot - new_h / 2)
        new_x2 = int(cx_rot + new_w / 2)
        new_y2 = int(cy_rot + new_h / 2)

        # 裁切區域不要超出圖片邊界
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(rotated.width, new_x2)
        new_y2 = min(rotated.height, new_y2)

        cropped = rotated.crop((new_x1, new_y1, new_x2, new_y2))
        return angle, cropped


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

    # 初始化 detector 並偵測
    from detector import FaceDetector
    detector = FaceDetector()
    boxes, probs, landmarks_all = detector.detect(img)

    # 沒有偵測到任何臉
    if boxes is None or len(boxes) == 0:
        print("No faces detected.")
        exit(0)

    # 初始化 aligner
    aligner = FaceAligner(expand_ratio=1.4)
    draw = ImageDraw.Draw(img)

    # 使用內建字體，可調整大小
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except:
        font = ImageFont.load_default()

    for idx, box in enumerate(boxes):
        box = boxes[idx]
        landmarks = landmarks_all[idx]
        angle, cropped = aligner.align_face(img, box, landmarks)

        # 畫原圖上的左右眼與連線
        left_eye = tuple(landmarks[0])
        right_eye = tuple(landmarks[1])
        r = 3
        draw.ellipse((left_eye[0]-r, left_eye[1]-r, left_eye[0]+r, left_eye[1]+r), fill="red")
        draw.ellipse((right_eye[0]-r, right_eye[1]-r, right_eye[0]+r, right_eye[1]+r), fill="red")
        draw.line([left_eye, right_eye], fill="blue", width=2)

        # 計算文字位置，確保不超出圖片
        text = f"{angle:.2f}°"
        print(angle)
        try:
            bbox = draw.textbbox((0,0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = font.getsize(text)
            # Pillow 舊版 fallback
            text_w, text_h = 50, 15
        text_x = max(0, int(box[0]))
        text_y = max(0, int(box[1]-text_h-2))
        draw.text((text_x, text_y), text, fill="yellow", font=font)

        # 顯示裁切後的人臉
        cropped.show(title=f"Face #{idx}")

    print(f"Detected {len(boxes)} faces")
    img.show()