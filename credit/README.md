# 🛡️ Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

> A real-time machine learning web application that detects fraudulent credit card transactions with high accuracy using Logistic Regression and Random Forest classifiers.

---

## 📌 Table of Contents
- [About the Project](#-about-the-project)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [ML Pipeline](#-ml-pipeline)
- [Model Performance](#-model-performance)
- [Features](#-features)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [What I Learned](#-what-i-learned)
- [Author](#-author)

---

## 📖 About the Project

Credit card fraud is a major financial threat worldwide. This project builds an end-to-end ML system that:

- Trains on **284,807 real-world transactions** (from Kaggle)
- Handles severe **class imbalance** (only 0.17% fraud cases) using SMOTE
- Compares multiple ML models with detailed metrics
- Provides a **live web app** for real-time fraud prediction with confidence scoring

---

## 🎥 Demo

### Fraud Detection Tab
> Enter transaction features → Get instant prediction with confidence %

### Model Comparison Tab
> Side-by-side accuracy, precision, recall, F1 + confusion matrices

### Data Insights Tab
> Dataset statistics, class distribution, amount analysis

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| ML Models | Scikit-learn (Logistic Regression, Random Forest) |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Web App | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## 📊 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Features:** 30 (Time, V1–V28 via PCA, Amount)
- **Target:** Class (0 = Legitimate, 1 = Fraudulent)
- **Fraud Rate:** 0.172% — highly imbalanced

> ⚠️ Due to confidentiality, original features are PCA-transformed (V1–V28). Only `Time` and `Amount` are in original form.

---

## 🔬 ML Pipeline

```
Raw Data (284,807 transactions)
        │
        ▼
Exploratory Data Analysis
        │
        ▼
SMOTE Oversampling  ◄── Handles class imbalance (better than undersampling)
        │
        ▼
Train / Test Split (80% / 20%)
        │
        ▼
Model Training
  ├── Logistic Regression
  └── Random Forest (100 estimators)
        │
        ▼
Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
        │
        ▼
Streamlit Web App Deployment
```

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | ~94% | ~94% | ~94% | ~94% |
| **Random Forest** | **~97%** | **~97%** | **~97%** | **~97%** |

> Results may vary slightly based on SMOTE random state. Random Forest consistently outperforms Logistic Regression on this dataset.

**Why these metrics matter for fraud detection:**
- **Recall** is critical — missing a fraud (false negative) is costly
- **Precision** matters too — flagging too many legit transactions frustrates users
- **F1 Score** balances both for an overall reliable model

---

## ✨ Features

- 🔍 **Real-time prediction** with fraud probability confidence score
- 🤖 **2 ML models** — choose between Logistic Regression and Random Forest
- 📊 **Model comparison dashboard** — metrics table + bar chart + confusion matrices
- 📈 **Data insights tab** — transaction stats, class distribution, amount analysis
- ⚖️ **SMOTE balancing** — industry-standard oversampling for imbalanced data
- 🎨 **Dark-themed UI** — clean, professional Streamlit interface

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
```

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/devanshdixit15/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `/credit` folder.

**4. Run the app**
```bash
streamlit run credit/app.py
```

**5. Open in browser**
```
http://localhost:8501
```

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── credit/
│   ├── app.py                          # Main Streamlit application
│   ├── credit.pkl                      # Saved model (pickle)
│   └── Credit Card Fraud Detection     # Project report & notebook
│       using Machine Learning.ipynb
│
├── requirements.txt                    # Python dependencies
└── README.md                           # You are here
```

---

## 📦 Requirements

```txt
streamlit
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
```

Or install directly:
```bash
pip install streamlit pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

---

## 💡 What I Learned

- Handling **highly imbalanced datasets** using SMOTE vs undersampling
- Why **Recall matters more than Accuracy** in fraud detection
- Building and comparing **multiple ML models** with proper evaluation metrics
- Deploying an ML model as a **production-ready web app** using Streamlit
- Interpreting **confusion matrices** and understanding false positives/negatives

---

## 🔮 Future Improvements

- [ ] Add XGBoost and compare with existing models
- [ ] Real-time CSV upload for batch prediction
- [ ] Deploy on Streamlit Cloud / Hugging Face Spaces
- [ ] Add SHAP values for model explainability
- [ ] Build REST API using FastAPI

---

## 👨‍💻 Author

**Devansh Dixit**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/devanshdixit15/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/devanshdixit15)

---

## ⭐ Show Your Support

If you found this project helpful, please consider giving it a **star ⭐** — it means a lot!

---

*Built with ❤️ by Devansh Dixit | GLA University, B.Tech CSE 2026*
