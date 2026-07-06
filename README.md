#  Skin Cancer Classification using Convolutional Neural Networks (CNN)

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Keras](https://img.shields.io/badge/Keras-3-red)
![License](https://img.shields.io/badge/License-MIT-green)

##  Overview

This project implements multiple **Convolutional Neural Network (CNN)** models for multiclass skin lesion classification using the **HAM10000** dataset.

Three CNN models were developed and compared based on:

- Prediction Accuracy
- Confidence Score
- Inference Time

The repository contains pre-trained models and an inference script for comparing their predictions on dermoscopic skin lesion images.

---

#  Features

- Multi-class skin lesion classification
- Comparison of three CNN models
- Confidence score prediction
- Inference time comparison
- Support for custom skin lesion images
- TensorFlow/Keras implementation

---

# 📂 Dataset

This project uses the **HAM10000 (Human Against Machine with 10,000 Training Images)** dataset.

Dataset:

https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

The dataset contains the following classes:

- Actinic Keratosis
- Basal Cell Carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented Benign Keratosis
- Seborrheic Keratosis
- Squamous Cell Carcinoma
- Vascular Lesion

> **Note**
>
> The dataset is **not included** in this repository due to its large size.

---

#  Project Structure

```text
SkinCancerClassification_Project/
│
├── images/
│   └── sample_image.jpg
│
├── models/
│   ├── skin_cancer_model.h5
│   ├── improved_skin_cancer_model.h5
│   └── final_skin_cancer_model.keras
│
├── screenshots/
│   └── prediction_output.png
│
├── compare_models.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

#  Installation

Clone the repository:

```bash
git clone https://github.com/Vansh-codi/SkinCancerClassification_Project.git
cd SkinCancerClassification_Project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

#  Running the Project

Place your test image inside:

```text
images/
```

Rename it to:

```text
sample_image.jpg
```

Run:

```bash
python compare_models.py
```

---

#  Model Comparison

| Model | Input Size | Epochs | Training Accuracy | Validation Accuracy |
|-------|-----------:|--------:|------------------:|--------------------:|
| Basic CNN | 180 × 180 | 25 | ~55% | ~39% |
| Improved CNN | 224 × 224 | 40 | **~62%** | **~39%** |
| Final CNN + Callbacks | 224 × 224 | 40 | ~56% | ~33% |

---

#  Example Prediction

![Prediction Output](screenshots/prediction_output.png)

---

#  Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- h5py
- Pillow
- Matplotlib
- Scikit-learn

---

#  Future Improvements

- Transfer Learning (EfficientNet, MobileNet)
- Hyperparameter Optimization
- Grad-CAM Visualization
- Streamlit or Flask Web Application
- Cloud Deployment

---

# ⚠️ Disclaimer

This project is intended for **educational and research purposes only** and should **not** be used as a substitute for professional medical diagnosis.

---

#  Author

**Vansh Jain**

GitHub: https://github.com/Vansh-codi
