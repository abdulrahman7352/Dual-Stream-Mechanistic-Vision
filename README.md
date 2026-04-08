---

# CIFAR-100 Results

## Results: Mechanistic Model (CIFAR-100)

### Robustness Under Stimulus Conditions

| Condition | Accuracy |
|-----------|----------|
| **Clear** | **56.39%** |
| **Blur** | **37.05%** |
| **Sharp** | **32.24%** |

---

### Validation Images (Mechanistic)

![Psychometric](results/mech/cifar100/psychometric_curve_mechanistic_cifar100.png)
![Confusion Matrix](results/mech/cifar100/confusion_matrix_mechanistic_cifar100.png)
![Grad-CAM](results/mech/cifar100/gradcam_mechanistic_cifar100.png)

---

## Results: Baseline ResNet50 (CIFAR-100)

### Robustness Under Stimulus Conditions

| Condition | Accuracy |
|-----------|----------|
| **Clear** | **82.42%** |
| **Blur** | **2.65%** |
| **Sharp** | **52.59%** |

---

### Validation Images (Baseline)

![Psychometric](results/baseline/validation/cifar100/psychometric_curve_baseline_cifar100.png)
![Confusion Matrix](results/baseline/validation/cifar100/confusion_matrix_baseline_cifar100.png)
![Grad-CAM](results/baseline/validation/cifar100/gradcam_baseline_cifar100.png)

---

## Head-to-Head Comparison (CIFAR-100)

| Condition | Mechanistic | Baseline |
|-----------|------------|----------|
| Clear | 56.39% | 82.42% |
| Blur | **37.05%** | **2.65%** |
| Sharp | 32.24% | 52.59% |

---

### Comparison Visualizations

![Psychometric](results/comparison/cifar100/psychometric_curve_compare_cifar100.jpeg)
![Confusion Matrix](results/comparison/cifar100/confusion_matrix_compare_cifar100.jpeg)
![Grad-CAM](results/comparison/cifar100/gradcam_compare_cifar100.jpeg)

---

## Key Insight

- Mechanistic model is **~14× more robust under blur**
- Baseline completely collapses (2.65%)
- Confirms **shape-based + predictive coding advantage**
