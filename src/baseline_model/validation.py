# ------------------------------------------------------------
# 1. Dynamic Blur Dataset for the Baseline
# ------------------------------------------------------------
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
        # Standard baseline only takes one input stream
        x = self.tf(img_blur)
        return x, self.labels[idx]

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
# 2. Run the Validations (Combined Curve & Baseline CM)
# ------------------------------------------------------------
# Make sure we use the correct weights filename that your script just saved
print("\n" + "="*60)
print("RUNNING BASELINE VALIDATIONS & COMBINED PLOTS")
print("="*60)

# 1. Load the Best Baseline Model
model.load_state_dict(torch.load("best_baseline_model.pth", weights_only=True))
model.eval()

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# --- TEST A: COMBINED PSYCHOMETRIC CURVE ---
print("\n--- Generating Combined Psychometric Curve ---")
blur_radii = [1, 2, 3, 4, 5, 6, 7, 8]
baseline_accuracies = []

# Hardcoded results from your previous Mechanistic Model run!
mech_accuracies = [0.7632, 0.6573, 0.6450, 0.5178, 0.3777, 0.2969, 0.2650, 0.2435]

for r in blur_radii:
    test_dataset = DynamicBaselineBlurDataset(test_data.data, test_data.targets, tf_test, blur_radius=r)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # We can reuse your existing evaluate() function since baseline takes 1 input (x, y)
    acc = evaluate(model, test_loader, device)
    baseline_accuracies.append(acc)
    print(f"Baseline Blur Radius {r}: Accuracy = {acc:.4f}")

# Plotting Both Curves Together
plt.figure(figsize=(8, 5))
plt.plot(blur_radii, mech_accuracies, marker='o', linestyle='-', color='blue', label='Mechanistic Model (Dual-Stream)')
plt.plot(blur_radii, baseline_accuracies, marker='X', linestyle='--', color='red', label='Baseline (Standard ResNet50)')

plt.title('Psychometric Degradation: Mechanistic vs Baseline')
plt.xlabel('Gaussian Blur Radius')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0)
plt.grid(True)
plt.legend()
plt.savefig('combined_psychometric_curve.png')
print("-> Saved 'combined_psychometric_curve.png'")
plt.show()

# --- TEST B: BASELINE CONFUSION MATRIX (At Blur=3) ---
print("\n--- Generating Baseline Confusion Matrix for Blur Radius 3 ---")
cm_dataset = DynamicBaselineBlurDataset(test_data.data, test_data.targets, tf_test, blur_radius=3)
cm_loader = DataLoader(cm_dataset, batch_size=128, shuffle=False, num_workers=2)

y_true, y_pred = get_baseline_preds(model, cm_loader, device)

print("\nBaseline Classification Report (Precision, Recall, F1-Score):")
report = classification_report(y_true, y_pred, target_names=classes)
print(report)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
# Using a Red color map to easily distinguish it from your blue Mechanistic matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix (Baseline ResNet50 on Blurred Images)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_baseline.png')
print("-> Saved 'confusion_matrix_baseline.png'")
plt.show()

print("\nAll Validations Complete! Check your folder for the generated images.")



# ------------------------------------------------------------
# 1. Self-Contained Grad-CAM Implementation
# ------------------------------------------------------------
class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hooks to grab the activations and gradients during forward/backward passes
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backward pass for the target class
        score = logits[0, target_class]
        score.backward()

        # Global average pooling of the gradients to get the weights
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Multiply activations by weights to get the CAM
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam) # We only care about features that positively contribute

        # Normalize between 0 and 1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam[0, 0].cpu().numpy(), target_class

# ------------------------------------------------------------
# 2. Helper to Overlay Heatmap on Image
# ------------------------------------------------------------
def show_cam_on_image(img_tensor, cam_mask):
    # Reverse the normalization we applied during preprocessing
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    img_np = img_tensor.cpu().numpy()
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    img_np = np.transpose(img_np, (1, 2, 0)) # Change to HxWxC

    # Resize the CAM mask to match the image size (224x224)
    cam_mask_resized = Image.fromarray(cam_mask).resize((224, 224), Image.Resampling.BILINEAR)
    cam_mask_resized = np.array(cam_mask_resized)

    # Apply colormap
    heatmap = plt.get_cmap('jet')(cam_mask_resized)[:, :, :3]

    # Overlay
    overlay = 0.5 * heatmap + 0.5 * img_np
    overlay = np.clip(overlay, 0, 1)

    return img_np, heatmap, overlay

# ------------------------------------------------------------
# 3. Main Execution
# ------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Grad-CAM on {device}...")

    # Load the same Baseline Model
    def get_baseline_model(num_classes=10):
        model = models.resnet50(weights=None) # We don't need to download weights again
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    model = get_baseline_model(num_classes=10).to(device)
    model.load_state_dict(torch.load("best_baseline_model.pth", map_location=device, weights_only=True))

    # In ResNet50, the last convolutional layer is layer4[-1]
    target_layer = model.layer4[-1]
    grad_cam = SimpleGradCAM(model, target_layer)

    # Load data and transforms
    tf_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_data = datasets.CIFAR10(root="./data", train=False, download=True)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    
    # 1. ADD THE BLUR HERE
    inference_blur = GaussianBlurTransform(radius=3)

    # 2. MATCH THE MECHANISTIC INDICES EXACTLY (Remove np.random.choice)
    indices = [3, 5, 8, 12, 18]  
    num_images = len(indices)

    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3 * num_images))
    fig.suptitle('Grad-CAM: Baseline ResNet50 (Blurred Input R=3)', fontsize=16)

    for i, idx in enumerate(indices):
        img_pil, true_label = test_data[idx]
        
        # 3. APPLY BLUR TO PIL IMAGE
        blurred_pil = inference_blur(img_pil)

        # 4. PASS THE BLURRED IMAGE TO THE TRANSFORM
        img_tensor = tf_test(blurred_pil).unsqueeze(0).to(device)

        # Generate the CAM
        cam, pred_label = grad_cam.generate_cam(img_tensor)

        # Process visual outputs
        original_img, heatmap, overlay = show_cam_on_image(img_tensor[0], cam)

        # Plot Original
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f"True: {classes[true_label]}")
        axes[i, 0].axis('off')

        # Plot Heatmap
        axes[i, 1].imshow(heatmap)
        axes[i, 1].set_title("Grad-CAM Heatmap")
        axes[i, 1].axis('off')

        # Plot Overlay
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Predicted: {classes[pred_label]}")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_baseline.png')
    print("-> Saved 'gradcam_baseline.png'")
    plt.show()