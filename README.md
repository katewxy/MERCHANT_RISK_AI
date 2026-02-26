# 🏦 Hybrid AI Risk Monitoring System

A production-style merchant risk control platform combining Machine Learning and rule-based engines to detect fraudulent transactions in real time.

Built with a clean modular architecture — not a notebook demo.

---

## 🚀 Live Demo

> Run locally with Streamlit — see setup below.

![Dashboard Preview](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![ML](https://img.shields.io/badge/Model-Logistic%20Regression-green)

---

## 🧠 System Architecture
```
MERCHANT_RISK_AI/
├── app/
│   └── dashboard.py          # Streamlit Risk Control Center
├── data/
│   └── raw/
│       └── creditcard.csv    # Kaggle fraud dataset (not included)
├── src/
│   ├── core/
│   │   ├── schema.py         # Data enrichment (merchant/customer IDs)
│   │   ├── governance.py     # Data cleaning & validation
│   │   └── features.py       # ML feature engineering
│   ├── models/
│   │   └── fraud_model.py    # Logistic Regression (class_weight=balanced)
│   ├── risk/
│   │   ├── rule_engine.py    # Rule-based risk scoring
│   │   ├── risk_engine.py    # Hybrid score = rule_risk + ml_probability
│   │   └── risk_metrics.py   # KPIs, rankings, trends
│   ├── services/
│   │   ├── pipeline.py       # End-to-end orchestration
│   │   └── analytics_service.py  # Clean API for dashboard
│   └── ai/
│       └── agent.py          # AI agent layer
└── requirements.txt
```

---

## ⚙️ How It Works

The system uses a **hybrid scoring approach**:

- **Rule Engine** — flags transactions based on amount thresholds and time-of-day patterns
- **ML Model** — Logistic Regression trained on Kaggle's credit card fraud dataset with `class_weight="balanced"` to handle severe class imbalance
- **Final Risk Score** = `rule_risk + ml_probability` (normalized)

---

## 📊 Dashboard Features

- KPI row: avg risk, fraud rate, high-risk count
- Risk score distribution histogram
- Daily risk trend (dual-axis)
- Merchant risk ranking table
- High-risk transaction table with full score breakdown
- Sidebar: merchant filter + risk threshold slider

---

## 🛠 Setup
```bash
# 1. Clone the repo
git clone https://github.com/katewxy/MERCHANT_RISK_AI.git
cd MERCHANT_RISK_AI

# 2. Download dataset
# Get creditcard.csv from Kaggle and place in data/raw/

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app/dashboard.py
```

---

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| Dashboard | Streamlit + Plotly |
| ML Model | Scikit-learn Logistic Regression |
| Data | Kaggle Credit Card Fraud Dataset |
| Language | Python 3.9+ |

---

## 📁 Dataset

This project uses the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
Download and place at `data/raw/creditcard.csv` before running.
