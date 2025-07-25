# 🛡 **ChurnShield – Customer Churn Prediction**

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**ChurnShield** is an ML-powered web app that predicts **customer churn** and provides **insights to retain customers**.  
Built with **Streamlit**, **XGBoost**, and **SHAP**, it offers a modern, interactive, and explainable AI experience.

---

## 🚀 **Live Demo**
👉 [**Try ChurnShield Now!**](https://share.streamlit.io/yourusername/churnshield)  
*(Replace with your Streamlit Cloud app link.)*

---

## ✨ **Features**
- **Customer Churn Prediction** using a trained XGBoost model.
- **Loyalty Score Gauge** for churn probability.
- **SHAP Explainability** – Feature importance and individual prediction explanation.
- **Visual EDA Dashboard** – 6 key churn trends with business insights.
- **Reset All Inputs** for quick refresh.
- Modern **pastel theme UI**.

---

## 📊 **Visual Insights**
The dashboard includes:
- Churn by **Gender**, **Contract**, and **Internet Service**.
- **Monthly vs Total Charges** trends.
- Correlation **heatmap**.
- Insights directly below each plot.

---

## ⚙️ **Tech Stack**
- **Frontend:** Streamlit + Plotly
- **ML Model:** XGBoost
- **Explainability:** SHAP
- **Data Analysis:** Pandas, NumPy, Matplotlib, Seaborn

---

## 🧠 **How It Works**
1. **User Input** – Enter customer details.
2. **Prediction** – Model outputs churn probability.
3. **Insights** – SHAP shows why the model predicted churn.

---

## 🔧 **Run Locally**
```bash
git clone https://github.com/yourusername/ChurnShield.git
cd ChurnShield
pip install -r requirements.txt
streamlit run streamlit_app.py
