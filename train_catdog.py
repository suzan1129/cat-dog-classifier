
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import RandomErasing
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------- Dataset ----------------
class CatDogDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images, self.labels = [], []
        for idx, name in enumerate(["Cat", "Dog"]):
            cls_dir = os.path.join(data_dir, name)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Class folder not found: {cls_dir}")
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.images.append(os.path.join(cls_dir, img_name))
                    self.labels.append(idx)

    def __len__(self): return len(self.images)

    def __getitem__(self, i):
        img = Image.open(self.images[i]).convert("RGB")
        y = self.labels[i]
        if self.transform: img = self.transform(img)
        return img, y

# ---------------- Model ----------------
def build_model():
    try:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    except Exception:
        model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)  # no Softmax; CE handles it
    return model

# ---------------- Utils ----------------
def plot_history(history, tag):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history["train_loss"], label="Train"); plt.plot(history["val_loss"], label="Val")
    plt.title(f"{tag} - Loss"); plt.xlabel("Epoch"); plt.legend()
    plt.subplot(1, 2, 2); plt.plot(history["train_acc"], label="Train"); plt.plot(history["val_acc"], label="Val")
    plt.title(f"{tag} - Accuracy"); plt.xlabel("Epoch"); plt.legend()
    os.makedirs("model", exist_ok=True)
    out = f"model/{tag.lower()}_training_history.png"
    plt.tight_layout(); plt.savefig(out); plt.close()
    print(f"[saved] {out}")

def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss_sum/total, correct/total

def train(model, train_loader, val_loader, device, epochs=20, lr=1e-3, tag="Normal"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # 加入 weight decay（一般會更穩定，訓練更快。）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs//3, 1), gamma=0.1)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_path = 0.0, f"model/resnet50_{tag.lower()}_best.pth"
    os.makedirs("model", exist_ok=True)

    for ep in range(1, epochs+1):
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"[{tag}] Epoch {ep}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * x.size(0)
            running_corrects += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss/total
        train_acc = running_corrects/total
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        scheduler.step()

        hist["train_loss"].append(train_loss); hist["train_acc"].append(train_acc)
        hist["val_loss"].append(val_loss);     hist["val_acc"].append(val_acc)

        print(f"Train  - loss: {train_loss:.4f}  acc: {train_acc*100:.2f}%")
        print(f"Val    - loss: {val_loss:.4f}  acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, best_path)
            print(f"[best] saved to {best_path} (val_acc={val_acc*100:.2f}%)")

    plot_history(hist, tag)
    return hist

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True, help="path/to/training_dataset")
    ap.add_argument("--val-dir",   required=True, help="path/to/validation_dataset")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--random-erasing", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tfm_base = [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    tfm_normal = transforms.Compose(tfm_base)
    tfm_erasing = transforms.Compose(tfm_base + [RandomErasing(p=0.5)])

    # Train normal model
    print("\n[1/2] Training NORMAL model")
    train_ds = CatDogDataset(args.train_dir, tfm_normal)
    val_ds   = CatDogDataset(args.val_dir, tfm_normal)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model_n = build_model().to(device)
    hist_n = train(model_n, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, tag="Normal")

    # Train random-erasing model (optional flag still trains both for comparison if provided)
    print("\n[2/2] Training RANDOM-ERASING model")
    train_ds_e = CatDogDataset(args.train_dir, tfm_erasing)
    val_ds_e   = CatDogDataset(args.val_dir, tfm_erasing)
    train_loader_e = DataLoader(train_ds_e, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader_e   = DataLoader(val_ds_e, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model_e = build_model().to(device)
    hist_e = train(model_e, train_loader_e, val_loader_e, device, epochs=args.epochs, lr=args.lr, tag="RandomErasing")

    # Save comparison for GUI
    comparison = {"normal": hist_n, "erasing": hist_e}
    os.makedirs("model", exist_ok=True)
    np.save("model/training_comparison.npy", comparison)
    print("[saved] model/training_comparison.npy")

if __name__ == "__main__":
    main()
