# app/embedder.py
# 載入 InceptionResnetV1 並計算 embedding
from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np
from PIL import Image

class FaceEmbedder:
    def __init__(self, device=None):
        import torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # 使用 facenet-pytorch 的預訓練模型 (vggface2)
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def preprocess(self, face_pil):
        """
        face_pil：PIL.Image 裁剪對齊的人臉
        回傳：歸一化的 torch 張量批次 (1,C,H,W)
        InceptionResnetV1 需要 160x160 的輸入。
        """
        face = face_pil.convert('RGB').resize((160,160))
        arr = np.asarray(face).astype(np.float32)
        # 歸一化：facenet-pytorch 期望像素值在 [-1,1] 範圍內
        arr = (arr / 127.5) - 1.0
        # HWC -> CHW
        tensor = torch.tensor(arr).permute(2,0,1).unsqueeze(0).to(self.device)
        return tensor

    def embed(self, face_pil):
        tensor = self.preprocess(face_pil)
        with torch.no_grad():
            embedding = self.model(tensor)  # shape (1,512)
        return embedding.cpu().numpy().flatten()
