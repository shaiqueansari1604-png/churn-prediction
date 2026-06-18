# 📉 Customer Churn Prediction

A machine learning pipeline to predict telecom customer churn using classical ML algorithms — with full EDA, preprocessing, model comparison, and inference support.

---

## 📁 Project Structure

```
churn-prediction/
├── data/                  # Raw dataset (not tracked in git)
├── notebooks/
│   └── eda.ipynb          # Exploratory Data Analysis
├── src/
│   ├── preprocess.py      # Data cleaning & feature engineering
│   ├── train.py           # Model training & evaluation
│   └── predict.py         # Inference on new data
├── models/                # Saved model artifacts (generated)
├── reports/               # Charts & evaluation outputs (generated)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your dataset
Place your CSV file at `data/telecom_churn.csv`.
The dataset should contain customer features and a `Churn` target column (0/1).

> **Recommended dataset:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### 4. Run EDA notebook
```bash
cd notebooks
jupyter notebook eda.ipynb
```

### 5. Train models
```bash
cd src
python train.py
```

### 6. Predict on new data
```bash
python predict.py --input data/new_customers.csv
```

---

## 🤖 Models Compared

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosted trees (usually best) |

Best model is auto-selected by **ROC-AUC** and saved to `models/`.

---

## 📊 Outputs

After running `train.py`, you get:
- `reports/feature_importance.png` — Top 10 predictive features
- `reports/churn_distribution.png` — Churn class balance
- `reports/correlation_heatmap.png` — Feature correlations
- `models/*.pkl` — Best trained model + scaler

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)

---

## 📝 License

MIT License — feel free to use and modify.
