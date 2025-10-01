
import os
import numpy as np
import torch
import torch.nn as nn
import random
from torchvision import models, transforms
from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

# -------------------------------
# Dataset
# -------------------------------
# 
# 
# 
# 
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images, self.labels = [], []
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Dataset folder not found: {data_dir}")
        for label, class_name in enumerate(["Cat", "Dog"]):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                # skip silently so the demo can still run
                continue
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.images.append(os.path.join(class_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# -------------------------------
# Model
# -------------------------------
def get_resnet50_model(num_classes: int = 2):
    """ResNet50 backbone with a 2-class linear head (no Softmax)."""
    try:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    except Exception:
        # for older torchvision versions
        model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def _adapt_state_dict_keys_for_linear_head(state_dict):
    """Support loading checkpoints saved when the head was `Sequential(Linear, Softmax)`.

    Those checkpoints typically store fc params under keys like 'fc.0.weight'.
    This function maps them to 'fc.weight' so we can load into a Linear-only head.
    """
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("fc.0."):
            new_state["fc." + k[len("fc.0."):]] = v  # fc.0.weight -> fc.weight
        else:
            new_state[k] = v
    return new_state

def _load_checkpoint_compat(model, ckpt_path, _device = None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)  # support plain state_dict or full checkpoint
    state = _adapt_state_dict_keys_for_linear_head(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected

# -------------------------------
# UI helpers
# -------------------------------
def load_show_images():
    transform = get_transforms(use_random_erasing=False)
    inference_dataset = CustomDataset("Dataset/inference_dataset", transform=transform)

    # 隨機找一張貓與一張狗
    cat_idx = random.choice([i for i, lbl in enumerate(inference_dataset.labels) if lbl == 0])
    dog_idx = random.choice([i for i, lbl in enumerate(inference_dataset.labels) if lbl == 1])
    # cat_idx = inference_dataset.labels.index(0)
    # dog_idx = inference_dataset.labels.index(1)

    if not cat_idx or not dog_idx:
        raise RuntimeError("Inference dataset 需同時包含 Cat/ Dog 範例。")
    cat_img, _ = inference_dataset[cat_idx]
    dog_img, _ = inference_dataset[dog_idx]

    # --------- 反正規化函式 ---------
    def denormalize(img_tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return img_tensor * std + mean

    cat_img = denormalize(cat_img).clamp(0, 1)
    dog_img = denormalize(dog_img).clamp(0, 1)

    # --------- 畫圖 ---------
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cat_img.permute(1, 2, 0))
    plt.title("Cat")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(dog_img.permute(1, 2, 0))
    plt.title("Dog")
    plt.axis("off")

    plt.tight_layout()
    plt.show()



def show_resnet_structure():
    model = get_resnet50_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("\nResNet50 Architecture (2-class head):")
    summary(model, (3, 224, 224))

def show_accuracy_comparison():
    """Plot accuracy comparison between normal and random-erasing models."""
    try:
        data = np.load("model/training_comparison.npy", allow_pickle=True).item()
        acc_normal = max(data["normal"]["val_acc"])
        acc_erasing = max(data["erasing"]["val_acc"])
        accuracies = {
            "Without Random Erasing": float(acc_normal),
            "With Random Erasing": float(acc_erasing),
        }
    except Exception as e:
        print(f"Warning: could not load comparison data: {e}\nUsing demo values.")
        accuracies = {"Without Random Erasing": 0.85, "With Random Erasing": 0.88}

    plt.figure(figsize=(8, 6))
    plt.bar(list(accuracies.keys()), list(accuracies.values()))
    plt.title("Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies.values()):
        plt.text(i, v + 0.01, f"{v:.2%}", ha="center")
    plt.show()

def get_best_model_path():
    """Pick the better checkpoint based on stored 'val_acc' in the checkpoint dicts."""
    try:
        res_normal = torch.load("model/resnet50_normal_best.pth", map_location="cpu")
        res_erasing = torch.load("model/resnet50_randomerasing_best.pth", map_location="cpu")
        acc_n = res_normal.get("val_acc", 0.0)
        acc_e = res_erasing.get("val_acc", 0.0)
        return "model/resnet50_randomerasing_best.pth" if acc_e > acc_n else "model/resnet50_normal_best.pth"
    except Exception as e:
        print(f"Warning: {e}. Falling back to normal checkpoint path.")
        return "model/resnet50_normal_best.pth"

def get_transforms(use_random_erasing: bool = False):
    tfms = [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if use_random_erasing:
        tfms.append(transforms.RandomErasing(p=0.5))
    return transforms.Compose(tfms)

# -------------------------------
# Inference
# -------------------------------
def inference_catdog(image_bgr_or_rgb_np):
    """Run inference on a numpy image (BGR or RGB), return label with confidence."""
    # Accept both BGR (OpenCV) or RGB (already) arrays
    img = image_bgr_or_rgb_np
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected HxWx3 image array.")
    # Heuristic: if looks like BGR (OpenCV), convert to RGB
    if img[..., 0].mean() > img[..., 2].mean():
        # very rough heuristic; safe either way
        img_rgb = img[..., ::-1]
    else:
        img_rgb = img

    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = tfm(img_rgb).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_resnet50_model().to(device)
    ckpt_path = get_best_model_path()
    try:
        # ckpt = torch.load(ckpt_path, map_location="cpu")
        # state = ckpt.get("model_state_dict", ckpt)
        # state = _adapt_state_dict_keys_for_linear_head(state)
        # missing, unexpected = model.load_state_dict(state, strict=False)
        missing, unexpected = _load_checkpoint_compat(model, ckpt_path, device)
        if missing or unexpected:
            print(f"(compat) missing: {missing}, unexpected: {unexpected}")
    except Exception as e:
        return f"Error loading model: {e}"

    # model.eval()
    # with torch.no_grad():
    #     x = x.to(device)
    #     logits = model(x)[0]
    #     probs = torch.softmax(logits, dim=0)
    #     pred = int(torch.argmax(probs).item())
    #     conf = float(probs[pred].item())

    model.eval()
    with torch.no_grad():
        x = x.to(device)
        logits = model(x)[0]  # 確保 logits 是 1-D Tensor (2 elements)
        probs = torch.softmax(logits, dim=0)
        
        # ----------------------------------------------------
        # 無法分類 判斷邏輯
        # ----------------------------------------------------
        
        # 1. 定義模糊判斷閾值 
        AMBIGUITY_THRESHOLD = 0.05
        
        # 2. 取得最高機率和次高機率
        # topk(probs, 2) 會返回 (值, 索引)
        top_probs = torch.topk(probs, 2)[0]
        max_prob = top_probs[0].item()
        second_max_prob = top_probs[1].item()

        # 3. 判斷是否為模糊結果
        class_names = ["Cat", "Dog"] # 假設 0=Cat, 1=Dog
        
        if (max_prob - second_max_prob) < AMBIGUITY_THRESHOLD:
            # 機率差距小於閾值，視為無法分類
            result_label = "Cannot Classify (Ambiguous Prediction)"
        else:
            # 決定類別
            pred_index = torch.argmax(probs).item()
            result_label = f"Predicted: {class_names[pred_index]} (Prob: {max_prob*100:.1f}%)"
            
    # 這裡將返回您處理好的結果字串
    return result_label

    # label = "Cat" if pred == 0 else "Dog"
    # return f"{label} ({conf*100:.1f}%)"
