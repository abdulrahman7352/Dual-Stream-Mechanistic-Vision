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
        F_trans = np.fft.fftshift(np.fft.fft2(gray))
        cy, cx = h//2, w//2
        Y, X = np.ogrid[:h, :w]
        mask = np.sqrt((X-cx)**2 + (Y-cy)**2) > self.cutoff*np.sqrt(cx**2+cy**2)
        F_trans *= mask
        img_back = np.real(np.fft.ifft2(np.fft.ifftshift(F_trans)))
        img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min() + 1e-6)
        rgb = np.stack([img_back]*3, axis=-1)
        return Image.fromarray((rgb*255).astype(np.uint8))

# ------------------------------------------------------------
# Datasets
# ------------------------------------------------------------
class BaselineDataset(Dataset):
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
        return self.tf(img), self.labels[idx]

class DynamicBaselineBlurDataset(Dataset):
    def __init__(self, imgs, labels, tf, blur_radius):
        self.imgs = imgs
        self.labels = labels
        self.tf = tf
        self.blur = GaussianBlurTransform(blur_radius)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.fromarray(self.imgs[idx])
        img_blur = self.blur(img)
        x = self.tf(img_blur)
        return x, self.labels[idx]

# ------------------------------------------------------------
# Model: Standard ResNet50
# ------------------------------------------------------------
def get_baseline_model(num_classes=100): # CHANGED TO 100
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------
def train_epoch(model, loader, device, epoch, opt): 
    model.train()
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

def get_baseline_preds(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Predicting Baseline"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return all_labels, all_preds

# ------------------------------------------------------------
# Grad-CAM Implementation
# ------------------------------------------------------------
class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        score = logits[0, target_class]
        score.backward()
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam[0, 0].cpu().numpy(), target_class

def show_cam_on_image(img_tensor, cam_mask):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_np = img_tensor.cpu().numpy()
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    img_np = np.transpose(img_np, (1, 2, 0))
    cam_mask_resized = Image.fromarray(cam_mask).resize((224, 224), Image.Resampling.BILINEAR)
    cam_mask_resized = np.array(cam_mask_resized)
    heatmap = plt.get_cmap('jet')(cam_mask_resized)[:, :, :3]
    overlay = 0.5 * heatmap + 0.5 * img_np
    overlay = np.clip(overlay, 0, 1)
    return img_np, heatmap, overlay

# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"BASELINE: ResNet50 on {device} (CIFAR-100)")

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

    # CHANGED TO CIFAR100
    full_train_data = datasets.CIFAR100(root="./data", train=True, download=True)
    test_data = datasets.CIFAR100(root="./data", train=False, download=True)
    classes = test_data.classes # Dynamically load all 100 classes

    # --- DATA SPLIT MATCHING MECHANISTIC MODEL (40k train, 10k val) ---
    shuffled_indices = torch.randperm(len(full_train_data)).tolist()
    train_indices = shuffled_indices[:40000]
    val_indices = shuffled_indices[40000:50000]

    train_data = full_train_data.data[train_indices]
    train_targets = [full_train_data.targets[i] for i in train_indices]

    val_data = full_train_data.data[val_indices]
    val_targets = [full_train_data.targets[i] for i in val_indices]

    train_set = BaselineDataset(train_data, train_targets, tf_train)
    val_set = BaselineDataset(val_data, val_targets, tf_test)
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2)

    model = get_baseline_model(num_classes=100).to(device)

    print("\n=== TRAINING BASELINE RESNET50 ===")
    best_acc = 0.0
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-3)

    for epoch in range(1, 31):
        current_lr = 1e-4 if epoch <= 10 else 5e-5
        for param_group in opt.param_groups:
            param_group['lr'] = current_lr

        loss, train_acc = train_epoch(model, train_loader, device, epoch, opt)

        if epoch % 2 == 0:
            val_acc = evaluate(model, val_loader, device)
            print(f"Epoch {epoch}: Loss={loss:.4f}, Train={train_acc:.4f}, Val={val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_baseline_model.pth")
                print(f" ---> Best baseline model saved with Val ACC: {best_acc:.4f}")
        else:
            print(f"Epoch {epoch}: Loss={loss:.4f}, Train={train_acc:.4f}")

    # --- FINAL EVALUATION ---
    print("\n" + "="*60)
    print("BASELINE RESULTS (Standard ResNet50)")
    print("="*60)
    print("Loading the best performing baseline model...")
    model.load_state_dict(torch.load("best_baseline_model.pth", map_location=device))

    clear_loader = DataLoader(BaselineTestDataset(test_data.data, test_data.targets, tf_test, "clear"), batch_size=64)
    blur_loader  = DataLoader(BaselineTestDataset(test_data.data, test_data.targets, tf_test, "blur"), batch_size=64)
    sharp_loader = DataLoader(BaselineTestDataset(test_data.data, test_data.targets, tf_test, "sharp"), batch_size=64)

    print(f"CLEAR ACC: {evaluate(model, clear_loader, device):.4f}")
    print(f"BLUR  ACC: {evaluate(model, blur_loader, device):.4f}")
    print(f"SHARP ACC: {evaluate(model, sharp_loader, device):.4f}")

    