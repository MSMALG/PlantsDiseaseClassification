# 🌿 Plant Disease Classification — CS435 Machine Learning Project

> Automated detection of 38 plant disease categories using classical machine learning and deep learning, with an interactive Streamlit diagnostic assistant.

---

## Project Overview

This repository contains the full machine learning pipeline for our CS435 project. We compare three models — EfficientNetB0 (deep learning), SVM, and Random Forest — on the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset), and deploy the results through a Streamlit application with an integrated LLM chatbot powered by Llama 3.3 70B via Groq.

| Model | Type | Accuracy | Macro F1 |
|---|---|---|---|
| EfficientNetB0 | Deep Learning | 98.40% | 0.98 |
| SVM (HOG + PCA) | Classical ML | 64.00% | 0.64 |
| Random Forest (HOG + PCA) | Classical ML | 42.42% | 0.40 |

---

## Repository Structure

```
├── Models.ipynb        # Full training pipeline (EDA → training → evaluation)
└── app.py              # Streamlit application (local version)
```

> **Model files** (`.keras`, `.pkl`) are not stored in this repository due to GitHub's 100 MB file size limit. They can be downloaded from Google Drive — see the setup instructions below.

---

## Running Locally

### 1. Clone this repository

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the model files

All trained model artifacts are available on Google Drive. Download them and place them in the same directory as `app.py`:

 **[Download Model Files from Google Drive]([https://drive.google.com/YOUR_FOLDER_LINK](https://drive.google.com/drive/folders/1hA-YSZWCgHWRSMeGT8STh4bC8CUGFkN_?usp=sharing))**

The following files are required:

```
best_model.keras        hog_scaler.pkl
class_names.pkl         hog_pca.pkl
svm_model.pkl           label_encoder.pkl
rf_model.pkl            dl_accuracy.pkl
                        svm_accuracy.pkl
                        rf_accuracy.pkl
```

> Alternatively, you can generate all these files yourself by running `Models.ipynb` end-to-end. Training was done on Kaggle with a Tesla T4 GPU — running on CPU is possible but significantly slower for the deep learning phases.

### 4. Set up your Groq API key

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 5. Run the app

```bash
streamlit run app.py
```

---

##  Model Pipeline Summary

### Classical ML (SVM & Random Forest)
1. Resize to 64 × 64, convert to grayscale
2. Extract HOG features → 1,764-dimensional vector per image
3. Normalize with `StandardScaler`
4. Compress with PCA → 497 dimensions (95% variance retained)
5. Train SVM (`C=10`, RBF kernel) and Random Forest (200 trees)

### Deep Learning (EfficientNetB0)
1. Resize to 224 × 224, apply EfficientNet-specific preprocessing
2. **Phase 1:** Train custom head only (backbone frozen), 10 epochs, lr=1e-3
3. **Phase 2:** Unfreeze top 30 backbone layers, fine-tune 15 epochs, lr=1e-5
4. Best weights saved automatically via `ModelCheckpoint`

---

##  Streamlit App Features

- Upload any leaf image (.jpg or .png)
- Choose between EfficientNetB0, SVM, or Random Forest
- Instant prediction with confidence score
- **Plant Advisor** chatbot for structured treatment advice via Llama 3.3 70B (Groq)
- Domain shift warning for outdoor/field photographs

---

##  Live Deployment

A deployed version of this app is available at:

**➡️ [Live Streamlit App](https://plantleavesclassification.streamlit.app/)**

The deployment uses a modified version of `app.py` that fetches `rf_model.pkl` from Google Drive at runtime to work around Streamlit Cloud's file size constraints. See the [Deployment Repository]([https://github.com/YOUR_USERNAME/YOUR_DEPLOYMENT_REPO](https://github.com/MSMALG/mloutputFinal/tree/main)) for the full deployment setup.

---

## 👥 Team

| Name | Student ID |
|---|---|
| Muzna Abdelgadir | 441211827 |
| Bedor Alharbi | 432205469 |
| Raghad Mesleh | 441203195 |
| Horiah Algofidi | 441203342 |

**Course:** CS435 — Machine Learning  
**Lecturer:** Dr. Renad Alsweed  
**Section:** 5500 — May 2026
