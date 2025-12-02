# app/aligner.py
# 用左右眼計算旋轉、旋轉並裁切：
# 先旋轉全圖、再裁切放大區域。
# 為了更精準，pipeline 稍後會在旋轉後對 rotated 圖片再次 run MTCNN
import numpy as np
from PIL import Image
import math

def rotate_image(img, angle, center=None):
    return img.rotate(angle, resample=Image.BICUBIC, center=center, expand=True)

class FaceAligner:
    def __init__(self, expand_ratio=1.4):
        """
        expand_ratio: 在原 bounding box 的基礎上擴大幾倍去包含更多頭部
        """
        self.expand_ratio = expand_ratio

    def _get_rotate_angle(self, left_eye, right_eye):
        # 角度（度數）；正值 -> 逆時針旋轉
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = math.degrees(math.atan2(dy, dx))
        # 希望以該角度的負值旋轉，使眼睛水平。
        return -angle

    def align_face(self, pil_image, box, landmarks):
        """
        pil_image: PIL.Image
        box: [x1,y1,x2,y2] bounding box
        landmarks: 5x2 array, landmarks[0]=left_eye, [1]=right_eye, etc (as facenet-pytorch returns)
        returns: cropped_aligned_face (PIL.Image)
        """
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        angle = self._get_rotate_angle(left_eye, right_eye)

        # 將整個影像繞盒子中心旋轉。
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        rotated = rotate_image(pil_image, angle, center=(cx, cy))

        # 在原始影像座標系中計算擴展框，然後變換座標。
        w = (x2 - x1)
        h = (y2 - y1)
        new_w = w * self.expand_ratio
        new_h = h * self.expand_ratio

        new_x1 = cx - new_w / 2
        new_y1 = cy - new_h / 2
        new_x2 = cx + new_w / 2
        new_y2 = cy + new_h / 2

        # 旋轉影像並設定 expand=True 後，影像尺寸發生變化，中心位置也發生了移動。
        # 更簡單的方法：對旋轉後的圖像重新運行 MTCNN 以優化裁剪（我們將在流程中完成）。
        # 但對於快速裁剪，我們可以將座標裁剪到旋轉後的尺寸​​：
        rw, rh = rotated.size
        new_x1 = max(0, int(new_x1))
        new_y1 = max(0, int(new_y1))
        new_x2 = min(rw, int(new_x2))
        new_y2 = min(rh, int(new_y2))

        cropped = rotated.crop((new_x1, new_y1, new_x2, new_y2))
        return cropped
