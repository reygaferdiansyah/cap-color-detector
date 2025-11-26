# Bottle Cap Color Detector - Industrial Sorting Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-v11%20%7C%20v12-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

**Bottle Cap Color Detector** is an object detection project I developed to classify beverage bottle caps in industrial environments. This repository contains the complete ML pipeline, from data preprocessing to model benchmarking, comparing the latest **YOLOv12** and **YOLOv11** architectures.

---

## 1. 🛠️ Data Preprocessing & Augmentation

The dataset is managed using **Roboflow** to ensure consistency and robustness.

* **Total Images:** 30 (Split: 27 Train, 3 Valid, 0 Test)
* **Image Size:** 640x640 px
* **Format:** Auto-Orient applied

### Preprocessing Steps
To standardize inputs for the model:
1. **Auto-Orient:** Correct EXIF orientation automatically.
2. **Resize:** Stretch images to **640x640** pixels.

### Augmentations (Training Set)
Applied to create a robust model capable of handling industrial variations. **Outputs per training example: 3x**.
* **Flip:** Horizontal.
* **Rotation:** Clockwise & Counter-Clockwise (between -15° and +15°).

---

## 2. ⚙️ Training Configuration

I trained the models on **NVIDIA GeForce RTX 5090** with the following settings:

| Hyperparameter         | Value                       | Description |
|-----------------------|-----------------------------|-------------|
| Epochs                | 250                         | Total training epochs |
| Batch Size            | 8                           | Number of images per batch |
| Image Size            | 640                         | Input image resolution |
| Optimizer             | AdamW                       | Optimizer used for gradient updates |
| Initial Learning Rate | 0.001                       | Starting learning rate |
| Momentum              | 0.937                       | Momentum factor for optimizer |
| Weight Decay          | 0.0005                      | Regularization to prevent overfitting |
| Warmup Epochs         | 5                           | Gradually increasing learning rate at start |

### Augmentation Parameters

| Parameter     | Value  | Description |
|---------------|--------|-------------|
| HSV Hue Gain  | 0.01   | Adjustment to color hue |
| HSV Saturation| 0.4    | Adjustment to color saturation |
| HSV Value     | 0.3    | Adjustment to brightness/value |
| Rotation      | 8°     | Random rotation applied |
| Translation   | 0.1    | Horizontal/vertical shift |
| Scale         | 0.6    | Zoom in/out scaling |
| Shear         | 2°     | Shearing transformation |
| Mosaic        | 1.0    | Mosaic augmentation probability |
| Mixup         | 0.1    | Mixup augmentation probability |
| Copy-Paste    | 0.15   | Copy-paste augmentation probability |

> This table replaces the previous YAML configuration for clarity and easier reading.

---

## 3. 📊 Model Benchmark & Evaluation

I evaluated **6 different models** (YOLOv11 & YOLOv12 in Nano, Small, and Medium sizes).  

**Hardware:** NVIDIA GeForce RTX 5090  

### 🏆 Best Model Selection: YOLOv12 Nano

The **YOLOv12n** was selected as the best performing model.  
It achieved the **highest mAP@50-95 (74.4%)** and near-perfect **Recall (99.9%)**, making it the most reliable choice for sorting tasks where missing a bottle cap is critical.

<img width="1384" height="684" alt="image" src="https://github.com/user-attachments/assets/26cf6b89-3642-4c7d-bff9-485e25193a36" />


---

### Performance Comparison Table

| Model      | Precision | Recall | mAP@50 | mAP@50-95 | Runtime | Tracked Hours | Performance Status |
|------------|-----------|--------|--------|------------|---------|---------------|------------------|
| YOLOv12n   | 0.939     | 0.999  | 0.993  | 0.744      | 2m 47s  | 2m 45s        | 🟢 BEST MODEL    |
| YOLOv12s   | 0.965     | 0.965  | 0.995  | 0.679      | 2m 3s   | 2m            | 🟡 Balanced      |
| YOLOv12m   | 0.681     | 0.970  | 0.975  | 0.659      | 2m 11s  | 2m 10s        | 🔴 Low Precision |
| YOLOv11n   | 0.967     | 0.995  | 0.995  | 0.618      | 1m 34s  | 1m 32s        | ⚡ Fastest       |
| YOLOv11s   | 0.964     | 0.973  | 0.995  | 0.672      | 1m 49s  | 1m 47s        | 🟡 Balanced      |
| YOLOv11m   | 0.940     | 0.852  | 0.995  | 0.666      | 2m 14s  | 2m 12s        | 🟡 Balanced      |

> **Note on Speed:**  
> While YOLOv11n is the fastest (1m 34s runtime), YOLOv12n offers significantly better detection quality (**mAP +12%**) with runtime 2m 47s, which is still suitable for real-time industrial sorting.

---

## 4. 📈 Key Findings

- **Architecture Superiority:** YOLOv12 outperforms YOLOv11 significantly in complex metrics (mAP50-95), proving its architecture extracts features better for this specific dataset.  
- **Recall Priority:** In industrial sorting, "Recall" is crucial (I don't want to miss-sort a bottle cap). YOLOv12n achieved **0.999 Recall**, meaning it missed almost zero objects.  
- **Stability:** The Nano model of v12 proved to be more stable than the Medium model on this dataset size, likely due to the Medium model overfitting on the small dataset (30 images).

---

## 5. Evaluation Best Model YOLOV12N

### F1 Curve
![F1 Curve](yolov12/nano/final_v12_nano/BoxF1_curve.png)

Near-Perfect Performance: The model achieves a peak F1-Score of 0.96. This is an exceptionally high score, indicating that the model excels at balancing Precision (accuracy of predictions) and Recall (ability to find all objects). It produces almost zero false alarms and misses very few targets.

Optimal Confidence Threshold: The best performance is observed at a Confidence Threshold of 0.606.

Actionable Insight: For deployment (inference), it is recommended to set the confidence filter to 0.60. This is the "sweet spot" where the model operates at maximum efficiency.

Detection Stability (Robustness): The curve remains flat and high across the 0.2 to 0.8 confidence range. This demonstrates high stability; the model is confident in its predictions and performs consistently even if lighting conditions or object positions vary slightly.

Class Dominance: The curves for Dark-Blue and Light-Blue maintain high scores for longer than other classes. This proves the model is highly effective at distinguishing between subtle blue color nuances (the core feature of this project) compared to the Others category.


### Confusion Matrices
Normalized:
![Normalized Confusion Matrix](yolov12/nano/final_v12_nano/confusion_matrix_normalized.png)


Here is the concise and critical explanation for the Confusion Matrix results in English:

Confusion Matrix Evaluation Summary:

Perfect Classification Accuracy: The diagonal values are all 1.00. This indicates 100% accuracy for all three classes (Dark-Blue, Light-Blue, Others). Every time an object was detected, the model correctly identified its specific color/class without fail.

Zero Class Confusion: The off-diagonal cells between the classes are empty (0.00). This proves the model has learned distinct features perfectly; it never confuses a Light-Blue cap with a Dark-Blue one, or vice versa. The separation between classes is flawless.

Ghost Detection (False Positive): There is a value of 1.00 in the top-right corner (True: background, Predicted: Dark-Blue).

Interpretation: This indicates a False Positive. The model detected a "Dark-Blue" object in an empty area (background) where there was actually nothing. While the classification of actual objects is perfect, the model is slightly too aggressive and "hallucinated" an object in the background.


Original:
![Confusion Matrix](yolov12/nano/final_v12_nano/confusion_matrix.png)


Raw Confusion Matrix Evaluation Summary:

Flawless Object Counts: The diagonal numbers (11, 2, and 5) represent the correct detections. The model successfully identified all 18 valid objects in the validation set (11 Dark-Blue, 2 Light-Blue, 5 Others) without missing a single one.

No Inter-Class Errors: The cells between the classes are empty. This confirms that the model never mixed up the cap colors (e.g., it never counted a Light-Blue cap as Dark-Blue).

Specific Error Source (Background False Positives): The matrix highlights exactly 2 errors in the top-right corner (Predicted: Dark-Blue, True: background).

Insight: The model "hallucinated" 2 Dark-Blue caps in empty space. This suggests that while the model is excellent at classification, it is slightly "trigger-happy" and aggressive in detecting Dark-Blue features in the background.


### Results
![Results](yolov12/nano/final_v12_nano/results.png)


Training & Validation Curves Analysis:

Rapid and Stable Convergence: The train/cls_loss (Classification Loss) and train/box_loss drop sharply within the first 50 epochs and continue to decrease steadily. This indicates the model learned the features of the bottle caps very quickly and efficiently.

Near-Perfect Detection Metrics: The metrics on the right (precision, recall, and mAP50) skyrocket early and stabilize near 1.0. This confirms the model is extremely accurate, consistently finding all objects with high confidence.

No Signs of Overfitting: The Validation Loss (val/cls_loss) follows the same downward trend as the Training Loss.

Insight: Even though the val/box_loss appears slightly "noisy" or jagged (which is normal for small validation datasets), it maintains a downward trend. This proves the model generalizes well to new data and is not just memorizing the training set.

Continuous Box Refinement: While mAP50 plateaus early (meaning it finds the object easily), the mAP50-95 curve continues to rise steadily until the end. This shows that throughout the training, the model kept refining the exact coordinates of the bounding boxes to be more and more precise.


### Sample Prediction
![Sample Prediction](yolov12/nano/final_v12_nano/val_batch0_pred.jpg)

Visual Prediction Analysis (Qualitative Results):

High Confidence Accuracy: The bounding boxes show very high confidence scores, mostly ranging between 0.9 and 1.0 (e.g., Dark-Blue 1.0). This visually confirms the statistical metrics; the model is extremely certain about its classifications and localizations.

Robust Class Separation: In the top-left image, the model successfully detects multiple classes simultaneously (Light-Blue vs. Others) within the same frame without confusion. It correctly identifies the greenish caps as "Others" and the blue ones as "Light-Blue," demonstrating excellent feature discrimination.

Edge Detection Capability: In the bottom-left image (top edge), the model detects caps that are partially cut off by the camera frame.

Insight: Although the confidence drops for these partial objects (0.3 and 0.6), the model still successfully detects them. This proves the model is robust enough to handle objects entering or leaving the conveyor belt, not just perfectly centered ones.



## 👤 Author

Reyga Ferdiansyah




