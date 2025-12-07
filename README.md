# 架構
```
face-recognition/
├── images/
│   ├── input       # 待辨識圖
│   └── register    # 登錄人臉
├── app/
│   ├── __init__.py # 套件入口
│   ├── detector.py # 臉部偵測
│   ├── aligner.py  # 臉部對齊
│   ├── embedder.py # 特徵向量
│   ├── db.py       # 資料存取
│   ├── pipeline.py # 整體流程
│   └── recognizer.py # 人臉比對
├── register_faces.py   # 建立資料庫
├── recognize_faces.py  # 執行辨識
├── result              # 輸出結果
├── face_db.sqlite      # 臉部資料庫
└── requirements.txt    # 套件需求
```

# 環境建置說明
### 1.建立一個 Python 3.9 的虛擬環境 (使用 conda)
Python 版本不限制，只要與套件相容就行
```
conda create -n face_env python=3.9
conda activate face_myenv
```
### 2.一次性安裝所有套件
```
pip install -r requirements.txt
```
或著也可以直接列出
```
pip install facenet-pytorch torchvision torch pillow numpy opencv-python scikit-learn sqlalchemy
```

# 操作流程
1. 環境建置
2. 將要登錄的人臉放入 images/register，一人一張，檔名即人名
3. 執行 register_faces.py ，建立資料庫
4. 將要辨識的合照放入 images/input，或是專案中隨便一個地方
5. 將 recognize_faces.py 中 INPUT_IMAGE 設定為要辨識的合照路徑
6. 執行 recognize_faces.py 進行辨識
7. 於 result 查看輸出結果，包括各人臉的截圖 & 前三名相似