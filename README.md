# 🐱🐶 Cat–Dog Classifier (ResNet50 + PyQt5)

## 📌 專案介紹
本專案是一個基於 **PyTorch** 的深度學習應用，使用 **ResNet50** 模型進行 **貓狗影像分類**，並整合 **PyQt5 GUI** 提供互動介面。  
使用者可以透過圖形化介面載入圖片、查看模型結構、比較訓練結果，並對單張影像進行推論。

---

## 🚀 功能
- **Load Image**：選擇單張圖片並顯示於右側預覽框  
- **Show Image (Dataset Grid)**：展示資料集影像（示意用）  
- **Show Model Structure**：印出 ResNet50 結構（最後層改為二分類）  
- **Show Comparison**：顯示有/無 Random-Erasing 訓練模型的準確率比較圖  
- **Inference**：對已載入圖片推論結果（Cat / Dog）

---

## 🛠 環境需求
- Python 3.9+  
- 建議建立虛擬環境 (`venv` 或 `conda`)  

### 安裝套件
```bash
pip install -r requirements.txt
```

### requirements.txt 
```
opencv-contrib-python==4.10.0.84
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
matplotlib==3.7.3
torchsummary==1.5.1
PyQt5==5.15.11
```

---

## 📂 專案結構
```
CatDogClassifier_Project/
├─ model/                         # 模型權重 (需自行放置)
│  ├─ resnet50_normal_best.pth
│  └─ resnet50_randomerasing_best.pth
├─ process & result/              # 訓練過程與成果
│  ├─ normal_training_history.png
│  ├─ randomerasing_training_history.png
│  └─ CatDog.ipynb
├─ dataset/                       # 原始資料集
│  ├─ inference_dataset/          
│  ├─ training_dataset/
│  └─ validation_dataset/                         
├─ CatDogClassifier.py            # 模型相關功能 
├─ main.py                        # GUI 主程式
├─ train_catdog.py                # 訓練腳本
├─ requirements.txt
└─ README.md
```

---

## 🔑 模型權重 (Model Weights)
專案需要以下 **已訓練好的 ResNet50 權重檔案** 才能進行推論：

- `model/resnet50_normal_best.pth`  
- `model/resnet50_randomerasing_best.pth`  

這些檔案因體積過大，**不會包含在 GitHub Repo**。  
請自行使用 `train_catdog.py` 進行訓練，並將產生的 `.pth` 檔案放置於 `model/` 資料夾中。

> ⚠️ 若未放置權重檔案，GUI 中的 Inference 功能將無法運作。

---

## 📥 資料集
整理為以下結構：
```
dataset/
├─ training_dataset/
│  ├─ Cat/
│  └─ Dog/
├─ validation_dataset/
│  ├─ Cat/
│  └─ Dog/
└─ inference_dataset/
   ├─ Cat/
   └─ Dog/
```

repo 中保留 `inference_dataset` 少量圖片作示範，完整資料集請使用者自行下載。

---

## 🧠 模型訓練
使用 `train_catdog.py` 訓練兩個版本：

```bash
# 無 Random-Erasing
python train_catdog.py   --train-dir dataset/training_dataset   --val-dir dataset/validation_dataset   --save-path model/resnet50_normal_best.pth

# 有 Random-Erasing
python train_catdog.py   --train-dir dataset/training_dataset   --val-dir dataset/validation_dataset   --random-erasing   --save-path model/resnet50_randomerasing_best.pth
```

訓練過程會輸出 **loss/accuracy 曲線** 圖檔。

---

## 🎮 使用方式
啟動 GUI：
```bash
python main.py
```

操作流程：
1. **Load Image** → 選擇一張圖片，會顯示在右側預覽框  
2. **Show Model Structure** → 查看 ResNet50 架構 (二分類版)  
3. **Show Comparison** → 顯示兩模型的驗證準確率比較圖  
4. **Inference** → 對已載入圖片推論，結果顯示於下方 Result 區域  

---

## 🔧 技術細節
- **Backbone**：ResNet50 (torchvision.models.resnet50)  
- **分類頭**：全連接層修改為二分類 + Softmax  
- **輸入尺寸**：224×224×3 (RGB)  
- **數據增強**：Random-Erasing 
- **優化器 / 損失函數**：Adam / CrossEntropyLoss  

---

## ⚖️ 授權
本專案僅作學術研究與個人學習用途；資料集授權依其官方來源。
