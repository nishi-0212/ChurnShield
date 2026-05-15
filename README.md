# 🛡️ ChurnShield — Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?logo=streamlit)](https://churnshieldbynishi.streamlit.app/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-green)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-purple)](https://shap.readthedocs.io)

**ChurnShield** is an end-to-end machine learning system that predicts customer churn in the telecom industry and explains *why* each prediction was made using SHAP explainability — enabling targeted, data-driven retention strategies.

👉 **[Try the Live App](https://churnshieldbynishi.streamlit.app/)**

---

## 📊 Model Performance

| Model                        | Accuracy | ROC-AUC | F1 (Churn) | Churn Recall |
|------------------------------|----------|---------|------------|--------------|
| Logistic Regression          | 80.1%    | 0.846   | 0.581      | 52.1%        |
| Random Forest                | 80.5%    | 0.847   | 0.583      | 51.3%        |
| **XGBoost (threshold=0.43)** ✅ | **75.5%** | **0.837** | **0.631** | **78.9%** |

> LR/RF achieve higher accuracy but miss ~48% of churners. XGBoost is deployed because **churn recall matters most** — catching a churner is more valuable than avoiding a false retention offer. Threshold tuned from default 0.5 to 0.43 to optimise F1 on the minority churn class.
>
> *Run `notebooks/ChurnShield_Training.ipynb` to reproduce exact numbers.*

---

## 🧠 How It Works

```
Customer Data → Feature Engineering → XGBoost Model → Churn Probability
                                                     ↓
                                           SHAP Explainer
                                                     ↓
                                     Claim-level Feature Attribution
```

1. **Input** — Enter customer demographics, services, and billing details
2. **Prediction** — XGBoost outputs a churn probability (0–100%)
3. **Loyalty Score** — Visual gauge showing retention likelihood
4. **SHAP Explanation** — Bar chart showing which features drove the prediction and in which direction

---

## ✨ Features

- **3-model comparison** — Logistic Regression, Random Forest, XGBoost
- **Feature engineering** — 4 domain-motivated features beyond raw data
- **SHAP explainability** — Global importance + per-prediction attribution
- **Interactive Streamlit UI** — Loyalty gauge, SHAP bar chart, clean sidebar inputs
- **Auto-calculated total charges** — No manual entry needed

---

## 📁 Project Structure

```
ChurnShield/
├── streamlit_app.py              ← Main Streamlit application
├── requirements.txt              ← Pinned dependencies
├── README.md
├── .streamlit/
│   └── config.toml              ← Theme configuration
├── models/
│   ├── churnshield_model.pkl    ← Trained XGBoost model
│   └── model_columns.pkl        ← Feature column order
├── notebooks/
│   └── ChurnShield_Training.ipynb ← Full training pipeline
└── assets/
    ├── eda_plots.png
    ├── model_comparison.png
    ├── confusion_matrix.png
    ├── roc_curves.png
    ├── shap_importance.png
    └── shap_beeswarm.png
```

---

## 🔧 Run Locally

```bash
git clone https://github.com/nishi-0212/ChurnShield.git
cd ChurnShield
pip install -r requirements.txt
streamlit run streamlit_app.py
```

To retrain the model:
1. Download [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Place CSV in the project root
3. Open and run `notebooks/ChurnShield_Training.ipynb`

---

## 📦 Tech Stack

| Layer | Tools |
|-------|-------|
| ML Models | XGBoost, Scikit-learn |
| Explainability | SHAP |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| App | Streamlit |
| Export | Joblib |

---

## 📈 Dataset

**IBM Telco Customer Churn** — [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 7,043 customers | 20 features | ~26.5% churn rate
- Features: demographics, services subscribed, contract type, billing info

---

## 👩‍💻 Author

**Nishi Vishwakarma**  
B.Tech ECE + AI, IGDTUW Delhi  
[GitHub](https://github.com/nishi-0212) · [LinkedIn](https://linkedin.com/in/nishi-vishwakarma)