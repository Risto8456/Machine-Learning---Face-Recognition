# app/detector.py
# MTCNN 初始化與 detect helper
from facenet_pytorch import MTCNN
import torch
from PIL import Image
import numpy as np

class FaceDetector:
    def __init__(self, device=None, keep_all=True):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # keep_all: True -> 回傳所有偵測到的臉
        self.mtcnn = MTCNN(keep_all=keep_all, device=self.device)

    def detect(self, pil_image):
        """
        使用 mtcnn.detect 回傳 boxes, probs, landmarks
        pil_image: PIL.Image
        """
        boxes, probs, landmarks = self.mtcnn.detect(pil_image, landmarks=True)
        # boxes: Nx4, landmarks: Nx5x2 (left_eye,right_eye,nose,left_mouth,right_mouth)
        return boxes, probs, landmarks
