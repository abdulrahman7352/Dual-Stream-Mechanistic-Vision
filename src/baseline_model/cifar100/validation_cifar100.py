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

    # --- TEST A: PSYCHOMETRIC CURVE ---
    print("\n--- Generating Psychometric Curve ---")
    blur_radii = [1, 2, 3, 4, 5, 6, 7, 8]
    baseline_accuracies = []

    for r in blur_radii:
        test_dataset = DynamicBaselineBlurDataset(test_data.data, test_data.targets, tf_test, blur_radius=r)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
        acc = evaluate(model, test_loader, device)
        baseline_accuracies.append(acc)
        print(f"Baseline Blur Radius {r}: Accuracy = {acc:.4f}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(blur_radii, baseline_accuracies, marker='X', linestyle='-', color='red', label='Baseline (ResNet50)')
    plt.title('Psychometric Degradation: Baseline')
    plt.xlabel('Gaussian Blur Radius')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.savefig('psychometric_curve_baseline.png')
    print("-> Saved 'psychometric_curve_baseline.png'")

    # --- TEST B: CONFUSION MATRIX ---
    print("\n--- Generating Baseline Confusion Matrix for Blur Radius 3 ---")
    cm_dataset = DynamicBaselineBlurDataset(test_data.data, test_data.targets, tf_test, blur_radius=3)
    cm_loader = DataLoader(cm_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    y_true, y_pred = get_baseline_preds(model, cm_loader, device)
    print("\nBaseline Classification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_true, y_pred, target_names=classes))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 16)) # Increased figure size to accommodate 100 classes
    sns.heatmap(cm, annot=False, cmap='Reds', xticklabels=classes, yticklabels=classes) # Turned off raw numbers to prevent 100x100 overlap
    plt.title('Confusion Matrix (Baseline ResNet50 on Blurred Images)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix_baseline.png')
    print("-> Saved 'confusion_matrix_baseline.png'")

    # --- TEST C: GRAD-CAM ---
    print("\n--- Running Grad-CAM on Baseline ---")
    target_layer = model.layer4[-1]
    grad_cam = SimpleGradCAM(model, target_layer)
    inference_blur = GaussianBlurTransform(radius=3)

    indices = [3, 5, 8, 12, 18]  
    num_images = len(indices)
    fig, axes = plt.subplots(num_images, 3, figsize=(10, 3 * num_images))
    fig.suptitle('Grad-CAM: Baseline ResNet50 (Blurred Input R=3)', fontsize=16)

    for i, idx in enumerate(indices):
        img_pil, true_label = test_data[idx]
        blurred_pil = inference_blur(img_pil)
        img_tensor = tf_test(blurred_pil).unsqueeze(0).to(device)

        cam, pred_label = grad_cam.generate_cam(img_tensor)
        original_img, heatmap, overlay = show_cam_on_image(img_tensor[0], cam)

        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(f"True: {classes[true_label]}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(heatmap)
        axes[i, 1].set_title("Grad-CAM Heatmap")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"Predicted: {classes[pred_label]}")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_baseline.png')
    print("-> Saved 'gradcam_baseline.png'")