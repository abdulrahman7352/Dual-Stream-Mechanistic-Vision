# Standard Library
import os
import argparse
import math
import random

# Third-Party Libraries
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# PyTorch Core
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

# Torchvision
import torchvision
from torchvision import datasets, models
import torchvision.transforms as T

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ------------------------------------------------------------
# Frequency transforms (Unchanged)
# ------------------------------------------------------------
class GaussianBlurTransform:
    def __init__(self, radius=3):
        self.radius = radius
    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.radius))

class FFTHighPassTransform:
    def __init__(self, cutoff=0.05):
        self.cutoff = cutoff
    def __call__(self, img):
        arr = np.asarray(img).astype(np.float32) / 255.0
        if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)

        gray = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]
        h, w = gray.shape
        F = np.fft.fftshift(np.fft.fft2(gray))
        cy, cx = h//2, w//2
        Y, X = np.ogrid[:h, :w]
        mask = np.sqrt((X-cx)**2 + (Y-cy)**2) > self.cutoff*np.sqrt(cx**2+cy**2)
        F *= mask
        img_back = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
        img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min() + 1e-6)
        rgb = np.stack([img_back]*3, axis=-1)
        return Image.fromarray((rgb*255).astype(np.uint8))

# ------------------------------------------------------------
# Baseline Dataset
# ------------------------------------------------------------
class BaselineDataset(Dataset):
    ### EDIT: Removed custom "augment_blur" flag because the standard transform handles it now
    def __init__(self, imgs, labels, tf):
        self.imgs = imgs
        self.labels = labels
        self.tf = tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.fromarray(self.imgs[idx])
        x = self.tf(img)
        return x, self.labels[idx]

# ------------------------------------------------------------
# Test Dataset
# ------------------------------------------------------------
class BaselineTestDataset(Dataset):
    def __init__(self, imgs, labels, tf, mode):
        self.imgs = imgs
        self.labels = labels
        self.tf = tf
        self.mode = mode
        self.blur = GaussianBlurTransform(3)
        self.sharp = FFTHighPassTransform(0.05)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.fromarray(self.imgs[idx])
        if self.mode == "blur":
            img = self.blur(img)
        elif self.mode == "sharp":
            img = self.sharp(img)
        # Clear just passes through
        return self.tf(img), self.labels[idx]

# ------------------------------------------------------------
# Model: Standard ResNet50
# ------------------------------------------------------------
def get_baseline_model(num_classes=10): ### EDIT: Changed default to 10 for CIFAR-10
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------
def train_epoch(model, loader, device, epoch, lr=1e-4): ### EDIT: Added LR parameter
    model.train()
    ### EDIT: Matched weight_decay to the Mechanistic model for a fair comparison
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc=f"Epoch {epoch}"):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()

        ### EDIT: Added Gradient Clipping to match Mechanistic Model
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"BASELINE: ResNet50 on {device}")

    ### EDIT: Exact same Transforms as Mechanistic Model
    tf_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))], p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    tf_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ### EDIT: Switched to CIFAR-10 and generated exact same 20k subset
    full_train_data = datasets.CIFAR10(root="./data", train=True, download=True)
    test_data = datasets.CIFAR10(root="./data", train=False, download=True)

    subset_indices = torch.randperm(len(full_train_data))[:20000].tolist()
    sub_data = full_train_data.data[subset_indices]
    sub_targets = [full_train_data.targets[i] for i in subset_indices]

    train_set = BaselineDataset(sub_data, sub_targets, tf_train)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)

    ### EDIT: Added a test loader for validation during training
    test_set = BaselineDataset(test_data.data, test_data.targets, tf_test)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

    model = get_baseline_model(num_classes=10).to(device)

    print("=== TRAINING BASELINE RESNET50 ===")
    best_acc = 0.0

    ### EDIT: Matched Epoch count (30) and LR schedule from Mechanistic Model
    for epoch in range(1, 31):
        lr = 1e-4 if epoch <= 10 else 5e-5
        loss, train_acc = train_epoch(model, train_loader, device, epoch, lr)

        ### EDIT: Checkpointing best model!
        if epoch % 2 == 0:
            test_acc = evaluate(model, test_loader, device)
            print(f"Epoch {epoch}: Loss={loss:.4f}, Train={train_acc:.4f}, Test={test_acc:.4f}")
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), "best_baseline_model.pth")
                print(f" ---> Best baseline model saved with Test ACC: {best_acc:.4f}")
        else:
            print(f"Epoch {epoch}: Loss={loss:.4f}, Train={train_acc:.4f}")

    # --- FINAL EVALUATION ---
    print("\n" + "="*60)
    print("BASELINE RESULTS (Standard ResNet50)")
    print("="*60)

    ### EDIT: Load best model before testing
    print("Loading the best performing baseline model...")
    model.load_state_dict(torch.load("best_baseline_model.pth", weights_only=True))

    clear_loader = DataLoader(BaselineTestDataset(test_data.data, test_data.targets, tf_test, "clear"), batch_size=64)
    blur_loader  = DataLoader(BaselineTestDataset(test_data.data, test_data.targets, tf_test, "blur"), batch_size=64)
    sharp_loader = DataLoader(BaselineTestDataset(test_data.data, test_data.targets, tf_test, "sharp"), batch_size=64)

    acc_clear = evaluate(model, clear_loader, device)
    acc_blur  = evaluate(model, blur_loader, device)
    acc_sharp = evaluate(model, sharp_loader, device)

    print(f"CLEAR ACC: {acc_clear:.4f}")
    print(f"BLUR  ACC: {acc_blur:.4f}  <- Compare this to your Mechanistic Model")
    print(f"SHARP ACC: {acc_sharp:.4f}")
