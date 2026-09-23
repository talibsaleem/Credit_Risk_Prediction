# 🏦 Real-Time Credit Risk & Default Prediction System with Explainable AI (XAI)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-228B22?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-8B008B?style=for-the-badge)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning and Explainable AI (XAI) solution designed to evaluate loan applications, estimate credit default probabilities, and provide transparent, auditable decision drivers using SHAP (SHapley Additive exPlanations) values.

---

## 📌 Business & Financial Impact
In retail banking, undetected loan defaults directly increase Non-Performing Assets (NPAs) and regulatory financial penalties. 
- **The Challenge:** Default cases in real-world data are inherently rare (imbalanced), and traditional "black-box" ML models fail strict regulatory compliance requirements (e.g., Fair Credit Reporting Rules) because they cannot justify rejections.
- **The Solution:** This system combines **SMOTE-resampled XGBoost** to catch high-risk borrowers with **SHAP interpretability layers** to generate automated, compliant explanation codes for credit officers.

---

## ⚡ Key Features
- 📊 **Exploratory Data Analysis (EDA):** Deep domain analysis on credit amount, loan duration, and age distribution risks.
- ⚙️ **Robust Preprocessing Pipeline:** Automatic categorical encoding, dummy feature alignment, and stratified splits.
- ⚖️ **Imbalance Management (SMOTE):** Addressed class imbalance using Synthetic Minority Over-sampling to boost minority default recall.
- 🤖 **Predictive Modeling:** Benchmark analysis between Random Forest and tuned **XGBoost Classifier**.
- 🔍 **Explainable AI (SHAP):** Global feature rankings and local customer-level waterfall plots showing exact positive/negative risk contributors.
- 🖥️ **Interactive Web Application:** Production-ready Streamlit dashboard for real-time risk scoring and visual auditability.

---

## 📊 Model Performance Comparison

| Model | Class Imbalance Handling | Default Recall (Class 1) | ROC-AUC Score | Primary Focus |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest** | SMOTE | 33% | ~0.78 | Baseline Model |
| **XGBoost Classifier** | **SMOTE** | **52%** | **~0.77** | **Optimal Risk Recall** |

> *Note: In credit scoring, higher **Recall** on Class 1 (Defaults) is prioritized over raw accuracy to minimize financial loss from bad debts.*

---

## 🛠️ Tech Stack
- **Core Language:** Python 3.11
- **Data Engineering:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
- **Model Explainability:** SHAP (TreeExplainer)
- **Web App / UI:** Streamlit
- **Visualization:** Matplotlib, Seaborn
- **Serialization:** Joblib

---

## 📁 Repository Structure
```text
Credit_Risk_Prediction/
├── data/
│   ├── credit_data.csv          # Raw German Credit Dataset
│   └── processed_data.npz       # Preprocessed Numpy arrays
├── models/
│   ├── credit_risk_xgb_model.pkl # Trained XGBoost Model
│   ├── feature_names.pkl        # Encoded Feature Mapping
│   ├── shap_summary.png         # SHAP Global Importance Plot
│   └── shap_customer_waterfall.png # Sample SHAP Explanation
├── download_data.py             # UCI Dataset Acquisition Script
├── 02_preprocessing.py          # Encoding & Stratification Script
├── 03_train_model.py            # SMOTE Resampling & Model Training
├── 04_explainability.py         # SHAP Evaluation Pipeline
├── app_gui.py                   # Streamlit Interactive Dashboard
└── README.md                    # System Documentation

## 🚀 How to Run Locally

1. Clone Repository & Setup Directory
     git clone [https://github.com/your-username/Credit_Risk_Prediction.git](https://github.com/your-username/Credit_Risk_Prediction.git)
    cd Credit_Risk_Prediction

2. Install Required Dependencies
    pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn shap streamlit joblib

3. Pipeline Execution (Optional - Pre-trained models included)
    python download_data.py
    python 02_preprocessing.py
    python 03_train_model.py
    python 04_explainability.py

4. Launch Streamlit Dashboard
    python -m streamlit run app_gui.py

👤 Author & Contact

Talib Saleem

Focus Areas: Machine Learning, Credit Risk Analytics, Explainable AI
