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
# Frequency transforms
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
        if arr.ndim == 2:
            gray = arr
        else:
            gray = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]

        h, w = gray.shape
        F_vals = np.fft.fftshift(np.fft.fft2(gray))
        cy, cx = h//2, w//2
        Y, X = np.ogrid[:h, :w]
        mask = np.sqrt((X-cx)**2 + (Y-cy)**2) > self.cutoff*np.sqrt(cx**2+cy**2)
        F_vals *= mask

        img_back = np.real(np.fft.ifft2(np.fft.ifftshift(F_vals)))
        img_back -= img_back.min()
        img_back /= img_back.max() + 1e-6

        rgb = np.stack([img_back]*3, axis=-1)
        return Image.fromarray((rgb*255).astype(np.uint8))

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class DualStreamDataset(Dataset):
    ### EDIT: Added fine_tf_test to init to allow clean blurring during dropout ###
    def __init__(self, imgs, labels, fine_tf, fine_tf_test, coarse_tf, blur_radius=3, fine_dropout_prob=0.0):
        self.imgs = imgs
        self.labels = labels
        self.fine_tf = fine_tf        # Augmented Transform
        self.fine_tf_test = fine_tf_test # Clean Transform
        self.coarse_tf = coarse_tf
        self.blur = GaussianBlurTransform(blur_radius)
        self.fine_dropout_prob = fine_dropout_prob

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = Image.fromarray(self.imgs[i])

        # 1. Coarse Stream: Always Blurred + Grayscale
        img_blur = self.blur(img)
        x_coarse = self.coarse_tf(img_blur)

        # 2. Fine Stream: Conditional Dropout
        ### EDIT: Logic simplified to avoid double-blurring (Blur + RandomCrop/Blur) ###
        if random.random() < self.fine_dropout_prob:
            # Dropout Active: Use CLEAN test transform on a BLURRED image
            x_fine = self.fine_tf_test(self.blur(img))
        else:
            # Normal: Use AUGMENTED transform
            x_fine = self.fine_tf(img)

        return x_coarse, x_fine, self.labels[i]

# ------------------------------------------------------------
# Model Components (ConvLSTM & DualBranchNet)
# ------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch):
        super().__init__()
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, 3, padding=1)

    def forward(self, x, state):
        if state is None:
            h = torch.zeros(x.size(0), self.hid_ch, *x.shape[2:], device=x.device)
            c = torch.zeros(x.size(0), self.hid_ch, *x.shape[2:], device=x.device)
        else:
            h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

class DualBranchNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        res18 = models.resnet18(weights="IMAGENET1K_V1")
        res18.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=4, bias=False)
        self.coarse_backbone = nn.Sequential(
            res18.conv1, res18.bn1, res18.relu, res18.maxpool,
            res18.layer1, res18.layer2, res18.layer3, res18.layer4
        )
        self.coarse_reduce = nn.Conv2d(512, 256, 1)

        res50 = models.resnet50(weights="IMAGENET1K_V1")
        self.fine_backbone = nn.Sequential(
            res50.conv1, res50.bn1, res50.relu, res50.maxpool,
            res50.layer1, res50.layer2, res50.layer3, res50.layer4
        )
        self.fine_reduce = nn.Conv2d(2048, 256, 1)

        self.coarse_lstm = ConvLSTMCell(256 + 256, 256)
        self.fine_lstm = ConvLSTMCell(256 + 256 + 256, 256)

        self.coarse_to_fine_pred = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 1))
        self.fine_to_coarse_feedback = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 1))

        self.fusion = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, 1))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Dropout(0.5), nn.Linear(512, num_classes))
        self.coarse_aux_head = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(256, num_classes))

    def forward(self, x_coarse, x_fine):
        c_feat = self.coarse_reduce(self.coarse_backbone(x_coarse))
        f_feat = self.fine_reduce(self.fine_backbone(x_fine))
        if c_feat.shape[-2:] != f_feat.shape[-2:]:
            c_feat = F.interpolate(c_feat, size=f_feat.shape[-2:], mode='bilinear', align_corners=False)

        h_coarse = h_fine = None
        for iteration in range(3):
            prediction = self.coarse_to_fine_pred(h_coarse[0]) if h_coarse is not None else torch.zeros_like(f_feat)
            error = f_feat - prediction
            if h_fine is not None:
                fine_feedback = self.fine_to_coarse_feedback(h_fine[0])
                fine_feedback = F.interpolate(fine_feedback, size=c_feat.shape[-2:], mode='bilinear', align_corners=False)
            else:
                fine_feedback = torch.zeros_like(c_feat)

            h_coarse = self.coarse_lstm(torch.cat([c_feat, fine_feedback], dim=1), h_coarse)
            h_fine = self.fine_lstm(torch.cat([f_feat, prediction, error], dim=1), h_fine)

        h_c, _ = h_coarse
        h_f, _ = h_fine
        if h_c.shape[-2:] != h_f.shape[-2:]:
            h_c = F.interpolate(h_c, size=h_f.shape[-2:], mode='bilinear', align_corners=False)

        fused = self.fusion(torch.cat([h_c, h_f], dim=1))
        return self.head(fused), self.coarse_aux_head(h_c)

# ------------------------------------------------------------
# Test Dataset
# ------------------------------------------------------------
class DualStreamTestDataset(Dataset):
    def __init__(self, imgs, labels, fine_tf, coarse_tf, mode):
        self.imgs = imgs
        self.labels = labels
        self.fine_tf = fine_tf
        self.coarse_tf = coarse_tf
        self.mode = mode
        self.blur = GaussianBlurTransform(3)
        self.sharp = FFTHighPassTransform(0.05)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.fromarray(self.imgs[idx])
        x_c = self.coarse_tf(self.blur(img))
        if self.mode == "clear":
            x_f = self.fine_tf(img)
        elif self.mode == "blur":
            x_f = self.fine_tf(self.blur(img))
        elif self.mode == "sharp":
            x_f = self.fine_tf(self.sharp(img))
        return x_c, x_f, self.labels[idx]

# ------------------------------------------------------------
# Training & Evaluation
# ------------------------------------------------------------
def train_epoch(model, loader, device, epoch, lr=1e-4):
    model.train()
    # EDIT 1: Increased weight_decay from 1e-4 to 5e-3 to prevent overfitting
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    for x_c, x_f, y in tqdm(loader, desc=f"Epoch {epoch}"):
        x_c, x_f, y = x_c.to(device), x_f.to(device), y.to(device)
        opt.zero_grad()
        logits, aux_logits = model(x_c, x_f)
        loss = loss_fn(logits, y) + 0.4 * loss_fn(aux_logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x_c, x_f, y in loader:
            x_c, x_f, y = x_c.to(device), x_f.to(device), y.to(device)
            logits, _ = model(x_c, x_f)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    ### EDIT: Defined distinct training and testing transforms ###
    fine_tf_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))], p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    fine_tf_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    coarse_tf = T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])

    full_train_data = datasets.CIFAR10(root="./data", train=True, download=True)
    test_data = datasets.CIFAR10(root="./data", train=False, download=True)

    # --- NEW LEAK-FREE DATA SPLIT ---
    # Shuffle all 50,000 training indices
    shuffled_indices = torch.randperm(len(full_train_data)).tolist()
    
    # Grab 20k for training, and the next 5k for validation
    train_indices = shuffled_indices[:20000]
    val_indices = shuffled_indices[20000:25000]

    # Extract the actual images and labels based on indices
    train_data = full_train_data.data[train_indices]
    train_targets = [full_train_data.targets[i] for i in train_indices]

    val_data = full_train_data.data[val_indices]
    val_targets = [full_train_data.targets[i] for i in val_indices]

    # Create Datasets
    train_set = DualStreamDataset(train_data, train_targets, fine_tf_train, fine_tf_test, coarse_tf, fine_dropout_prob=0.5)
    val_set = DualStreamDataset(val_data, val_targets, fine_tf_test, fine_tf_test, coarse_tf, fine_dropout_prob=0.0)
    
    # Create DataLoaders (Notice we don't need a test_loader for the training loop anymore!)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2)

    model = DualBranchNet(num_classes=10).to(device)

    print("=== TRAINING WITH DATA AUGMENTATION & FINE DROPOUT ===")

    best_acc = 0.0 # Track the best accuracy

    for epoch in range(1, 31):
        lr = 1e-4 if epoch <= 10 else 5e-5
        loss, train_acc = train_epoch(model, train_loader, device, epoch, lr)
        
        if epoch % 2 == 0:
            # Evaluate on VALIDATION set, completely blind to the Test set
            val_acc = evaluate(model, val_loader, device)
            print(f"Epoch {epoch}: Loss={loss:.4f}, Train={train_acc:.4f}, Val={val_acc:.4f}")

            # Save the best model based on Validation performance!
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_model.pth")
                print(f" ---> Best model saved with Val ACC: {best_acc:.4f}")
        else:
            print(f"Epoch {epoch}: Loss={loss:.4f}, Train={train_acc:.4f}")

    print("\n" + "="*60 + "\nTESTING DIFFERENT STIMULUS CONDITIONS\n" + "="*60)

    # EDIT 4: Load the best model before testing!
    print("Loading the best performing model for final evaluation...")
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))

    clear_loader = DataLoader(DualStreamTestDataset(test_data.data, test_data.targets, fine_tf_test, coarse_tf, "clear"), batch_size=64)
    blur_loader = DataLoader(DualStreamTestDataset(test_data.data, test_data.targets, fine_tf_test, coarse_tf, "blur"), batch_size=64)
    sharp_loader = DataLoader(DualStreamTestDataset(test_data.data, test_data.targets, fine_tf_test, coarse_tf, "sharp"), batch_size=64)

    print(f"CLEAR ACC: {evaluate(model, clear_loader, device):.4f}")
    print(f"BLUR  ACC: {evaluate(model, blur_loader, device):.4f} <- Goal: Robustness")
    print(f"SHARP ACC: {evaluate(model, sharp_loader, device):.4f}")

