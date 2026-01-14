# OCTMNIST-Retinal-Disease-Classification-Using-CNNs

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-89.33%25-success)

## 📌 Project Overview
Optical Coherence Tomography (OCT) is a critical imaging technique for diagnosing retinal diseases. This project implements a **Convolutional Neural Network (CNN)** to classify 109,000+ OCT retinal images into 4 distinct clinical categories.

By addressing significant class imbalances in the dataset using **weighted loss functions** and **dropout regularization**, the model achieved robust performance metrics, aiding in the automated detection of potentially blinding conditions.



## 🩺 Dataset: OCTMNIST
The dataset consists of **109,309 images** (28x28 grayscale) categorized into 4 classes:
1.  **CNV:** Choroidal Neovascularization
2.  **DME:** Diabetic Macular Edema
3.  **DRUSEN:** Multiple Drusen
4.  **NORMAL:** Healthy Retina

## 🏗️ Methodology

### Model Architecture
A custom CNN was designed to extract morphological features from the retinal scans:
* **Feature Extraction:** 3 blocks of Conv2d + BatchNorm + ReLU + MaxPool.
* **Regularization:** Dropout (p=0.5) applied to the fully connected layers to prevent overfitting.
* **Classification Head:** Linear layers mapping flattened features to 4 class logits.

### Handling Class Imbalance
Exploratory Data Analysis (EDA) revealed an imbalance in the class distribution (Normal/Drusen vs. CNV/DME). To mitigate bias:
* **Weighted Cross-Entropy Loss:** Penalized misclassifications of minority classes more heavily.
* **Evaluation:** Focused on **Macro F1-Score** rather than just accuracy to ensure fair performance across all diseases.

## 📊 Key Results

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **89.33%** |
| **Macro F1-Score** | **80.53%** |

> **Analysis:** The confusion matrix indicates that the model distinguishes well between pathological cases (CNV/DME) and Normal cases, with the weighted loss successfully improving recall on underrepresented classes.

## 🛠️ Tools & Technologies
* **Framework:** PyTorch
* **Data Handling:** MedMNIST, NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn (Confusion Matrices, Training Curves)
* **Metrics:** Precision, Recall, F1-Score, Accuracy

## 🚀 Usage

```bash
# Clone the repository
git clone [https://github.com/YourUsername/OCTMNIST-CNN.git](https://github.com/YourUsername/OCTMNIST-CNN.git)

# Install dependencies
pip install torch torchvision medmnist matplotlib

# Train the model
python train.py
