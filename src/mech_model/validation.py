# ------------------------------------------------------------
# 1. Dynamic Blur Dataset for the Psychometric Curve
# ------------------------------------------------------------
class DynamicBlurTestDataset(Dataset):
    def __init__(self, imgs, labels, fine_tf, coarse_tf, blur_radius):
        self.imgs = imgs
        self.labels = labels
        self.fine_tf = fine_tf
        self.coarse_tf = coarse_tf
        self.blur = GaussianBlurTransform(blur_radius)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.fromarray(self.imgs[idx])
        img_blur = self.blur(img)
        
        x_c = self.coarse_tf(img_blur)
        x_f = self.fine_tf(img_blur) # Testing on blurred images
        
        return x_c, x_f, self.labels[idx]

def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_c, x_f, y in tqdm(loader, desc="Predicting"):
            x_c, x_f, y = x_c.to(device), x_f.to(device), y.to(device)
            logits, _ = model(x_c, x_f)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return all_labels, all_preds

# ------------------------------------------------------------
# 2. Run the Validations
# ------------------------------------------------------------
if __name__ == "__main__":
    # Ensure device and transforms are still loaded from your main script
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # CIFAR-10 Class Names for the Confusion Matrix
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("\n" + "="*60)
    print("RUNNING MECHANISTIC VALIDATIONS (No retraining required)")
    print("="*60)

    # Load the best weights
    model = DualBranchNet(num_classes=10).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    # --- TEST A: PSYCHOMETRIC CURVE ---
    print("\n--- Generating Psychometric Curve ---")
    blur_radii = [1, 2, 3, 4, 5, 6, 7, 8]
    accuracies = []

    for r in blur_radii:
        test_dataset = DynamicBlurTestDataset(test_data.data, test_data.targets, fine_tf_test, coarse_tf, blur_radius=r)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
        
        acc = evaluate(model, test_loader, device)
        accuracies.append(acc)
        print(f"Blur Radius {r}: Accuracy = {acc:.4f}")

    # Plotting the Curve
    plt.figure(figsize=(8, 5))
    plt.plot(blur_radii, accuracies, marker='o', linestyle='-', color='b', label='Mechanistic Model')
    plt.title('Psychometric Degradation Curve (Human-Like Vision)')
    plt.xlabel('Gaussian Blur Radius')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.savefig('psychometric_curve_mechanistic.png')
    print("-> Saved 'psychometric_curve_mechanistic.png'")
    plt.show()

    # --- TEST B: CONFUSION MATRIX (At Blur=3) ---
    print("\n--- Generating Confusion Matrix for Blur Radius 3 ---")
    cm_dataset = DynamicBlurTestDataset(test_data.data, test_data.targets, fine_tf_test, coarse_tf, blur_radius=3)
    cm_loader = DataLoader(cm_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    y_true, y_pred = get_predictions(model, cm_loader, device)
    print("\nClassification Report (Precision, Recall, F1-Score):")
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Mechanistic Model on Blurred Images)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix_mechanistic.png')
    print("-> Saved 'confusion_matrix_mechanistic.png'")
    plt.show()
    
    print("\nValidations Complete!")



# ------------------------------------------------------------
# 1. Self-Contained Grad-CAM for DualBranchNet
# ------------------------------------------------------------
class DualStreamGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks to grab the activations and gradients
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, x_coarse, x_fine, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        # Dual-input Forward pass (unpacking logits and aux_logits)
        logits, _ = self.model(x_coarse, x_fine)
        
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
            
        # Backward pass for the target class
        score = logits[0, target_class]
        score.backward()
        
        # Global average pooling of the gradients to get the weights
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Multiply activations by weights to get the CAM
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam) # Keep only positive contributions
        
        # Normalize between 0 and 1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam[0, 0].cpu().numpy(), target_class

# ------------------------------------------------------------
# 2. Helper to Overlay Heatmap on Image
# ------------------------------------------------------------
def show_cam_on_image(img_pil, cam_mask):
    # Process original image for visualization
    img_pil = img_pil.resize((224, 224), Image.Resampling.BILINEAR)
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    
    # Resize the CAM mask to match the image size
    cam_mask_resized = Image.fromarray(cam_mask).resize((224, 224), Image.Resampling.BILINEAR)
    cam_mask_resized = np.array(cam_mask_resized)
    
    # Apply colormap (Jet)
    heatmap = plt.get_cmap('jet')(cam_mask_resized)[:, :, :3]
    
    # Overlay heatmap on original image
    overlay = 0.5 * heatmap + 0.5 * img_np
    overlay = np.clip(overlay, 0, 1)
    
    return img_np, heatmap, overlay

# ------------------------------------------------------------
# 3. Main Execution
# ------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Grad-CAM on Mechanistic Model ({device})...")

    # Load CIFAR-10 Test Data
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    test_data = datasets.CIFAR10(root="./data", train=False, download=True)

    # Recreate the exact transforms from your script
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
    
    # We test at Blur Radius 3 (where the baseline failed catastrophically)
    inference_blur = GaussianBlurTransform(radius=3)

    # Load the Mechanistic Model
    model = DualBranchNet(num_classes=10).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    # Target the final convolutional layer in the Fusion block
    target_layer = model.fusion[-1] 
    grad_cam = DualStreamGradCAM(model, target_layer)

    # Pick 5 indices to match your previous baseline visual test
    indices = [3, 5, 8, 12, 18] 
    num_images = len(indices)
    
    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3.5 * num_images))
    fig.suptitle('Grad-CAM: Mechanistic Model (Blurred Input R=3)', fontsize=16)

    for i, idx in enumerate(indices):
        img_pil, true_label = test_data[idx]
        
        # Apply the blur to the stimulus
        stimulus_pil = inference_blur(img_pil)

        # Generate Dual Inputs
        x_coarse = coarse_tf(stimulus_pil).unsqueeze(0).to(device)
        x_fine = fine_tf_test(stimulus_pil).unsqueeze(0).to(device)

        # Generate the CAM from the fused representation
        cam, pred_label = grad_cam.generate_cam(x_coarse, x_fine)
        
        # Overlay visual outputs
        stimulus_np, heatmap, overlay = show_cam_on_image(stimulus_pil, cam)
        
        # Plot Original Stimulus
        axes[i, 0].imshow(stimulus_np)
        axes[i, 0].set_title(f"True: {classes[true_label]}")
        axes[i, 0].axis('off')
        
        # Plot Heatmap
        axes[i, 1].imshow(heatmap)
        axes[i, 1].set_title("Fused Layer Heatmap")
        axes[i, 1].axis('off')
        
        # Plot Overlay
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Predicted: {classes[pred_label]}")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_mechanistic.png')
    print("-> Saved 'gradcam_mechanistic.png'")
    plt.show()
