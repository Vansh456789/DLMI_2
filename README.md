markdown
# Breast Ultrasound Image Classification using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blueviolet)](https://kaggle.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<div align="center">
  <img src="https://img.shields.io/badge/Status-Completed-success.svg" alt="Status"/>
  <img src="https://img.shields.io/badge/Accuracy-88%25-brightgreen.svg" alt="Accuracy"/>
  <img src="https://img.shields.io/badge/Malignant%20Recall-95%25-critical.svg" alt="Recall"/>
</div>

## 📋 Overview

This project implements a deep learning-based image classification system for breast ultrasound images using the **BUSI (Breast Ultrasound Images) dataset**. The model classifies ultrasound images into three categories:
- **Normal** (133 images)
- **Benign** (437 images)
- **Malignant** (210 images)

The research systematically investigates the impact of **class imbalance**—a common challenge in medical imaging datasets—and compares four imbalance mitigation strategies.

### 🎯 Key Achievements
- **Best Accuracy**: 88.0% (Data Augmentation)
- **Best Malignant Recall**: 95% (Class Weighting)
- **Weighted F1-Score**: 0.877 (Data Augmentation)
- **45% improvement** in cancer detection sensitivity over baseline

## 📊 Results Summary

| Method | Accuracy | Macro F1 | Weighted F1 | Malignant Recall |
|--------|----------|----------|-------------|------------------|
| Baseline | 76.0% | 0.710 | 0.753 | 50% |
| Class Weighting | 86.0% | 0.854 | 0.865 | **95%** |
| Oversampling | 84.0% | 0.833 | 0.845 | 93% |
| **Data Augmentation** | **88.0%** | **0.867** | **0.877** | 93% |
| Focal Loss | 75.0% | 0.739 | 0.750 | 88% |

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Kaggle account (for GPU access) OR local GPU with CUDA
- 16GB+ RAM recommended

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/vanshraj/cs23b1079-breast-ultrasound-classification.git
cd cs23b1079-breast-ultrasound-classification
2. Install dependencies

bash
pip install -r requirements.txt
3. Download the dataset

Download from Kaggle

Place in /kaggle/input/ directory (for Kaggle)

OR update DATA_DIR path in notebook for local use

4. Run the notebook

bash
jupyter notebook vansh_assignment\(1\).ipynb
📦 Dependencies
Create a requirements.txt file with:

text
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
ipykernel>=6.0.0
tqdm>=4.65.0
Pillow>=9.5.0
🧠 Model Architecture
ResNet18 Configuration
Base Architecture: ResNet18 (trained from scratch)

Input Size: 224×224×3

Output Classes: 3 (Normal, Benign, Malignant)

Total Parameters: 4,908,227

Training Parameters
Parameter	Value
Optimizer	Adam (lr=0.001)
Batch Size	32
Max Epochs	30
Early Stopping	Patience = 5
Hardware	NVIDIA Tesla P100
🔬 Imbalance Handling Techniques
1. Baseline
Standard cross-entropy loss without handling.

2. Class Weighting
Weights computed as inverse class frequencies:

Normal: 1.95

Benign: 0.59

Malignant: 1.24

3. Oversampling
WeightedRandomSampler for balanced batches.

4. Data Augmentation
Random horizontal flip (p=0.5)

Random rotation (±15°)

Color jitter

5. Focal Loss
Modified loss with γ=2 focusing on hard examples.

📈 Detailed Results
Baseline
text
              precision    recall  f1-score   support
Normal           0.83      0.81      0.82       134
Benign           0.63      0.79      0.70        63
Malignant        0.77      0.50      0.61        40
    accuracy                           0.76       237
Class Weighting
text
              precision    recall  f1-score   support
Normal           0.89      0.88      0.89       134
Benign           0.89      0.78      0.83        63
Malignant        0.76      0.95      0.84        40
    accuracy                           0.86       237
Oversampling
text
              precision    recall  f1-score   support
Normal           0.90      0.84      0.87       134
Benign           0.79      0.79      0.79        63
Malignant        0.76      0.93      0.83        40
    accuracy                           0.84       237
Data Augmentation (Best Overall)
text
              precision    recall  f1-score   support
Normal           0.90      0.90      0.90       134
Benign           0.89      0.79      0.84        63
Malignant        0.80      0.93      0.86        40
    accuracy                           0.88       237
Focal Loss
text
              precision    recall  f1-score   support
Normal           0.88      0.69      0.78       134
Benign           0.64      0.78      0.71        63
Malignant        0.64      0.88      0.74        40
    accuracy                           0.75       237
🎯 Training Dynamics
Method	Best Validation Accuracy	Epoch
Baseline	79.75%	7
Class Weighting	89.45%	23
Oversampling	88.61%	18
Data Augmentation	88.61%	16
Focal Loss	83.54%	21
💡 Key Insights
Clinical Significance
45% improvement in Malignant recall (50% → 95%)

False negative rate reduced from 50% to just 5%

Balanced performance across all classes with augmentation

Best Practices
Priority	Recommended Technique
Cancer Detection	Class Weighting
Overall Accuracy	Data Augmentation
Balanced Performance	Oversampling
🔄 Reproduction Guide
On Kaggle
Create new Kaggle notebook with GPU

Upload vansh_assignment(1).ipynb

Add BUSI dataset

Run all cells (30-45 minutes)

On Local Machine
bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
📝 Citation
bibtex
@misc{vansh2026breast,
    author = {Vansh Raj Singh},
    title = {Breast Ultrasound Image Classification using Deep Learning},
    year = {2026},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/vanshraj/cs23b1079-breast-ultrasound-classification}}
}
📄 License
MIT License - see LICENSE file.

👨‍💻 Author
Vansh Raj Singh (CS23B1079)

📧 Email: vansh.raj@domain.com

🔗 GitHub: @vanshraj

<div align="center"> <b>⭐ Star this repository if you find it useful! ⭐</b> <br> <i>Last Updated: March 2026</i> </div> ```
