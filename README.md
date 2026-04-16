# 🚚 Supply Chain Late Delivery Risk Predictor

An end-to-end data science project that predicts late delivery risk in global e-commerce supply chains using Machine Learning and generates AI-powered proactive recovery actions using GPT-4o-mini.

🔗 **Live App:** [Supply Chain Risk Predictor](https://https://xcesenzww4daknafc2osn9.streamlit.app/)

---

## 📌 Problem Statement

In global e-commerce, **late delivery** is the primary driver of customer churn and increased operational costs. Supply chain managers often lack the foresight to identify which orders are at risk of delay before they leave the warehouse, leading to reactive rather than proactive management.

---

## 🎯 Project Goals

1. **Predict** — Identify orders with high late delivery risk using Machine Learning
2. **Analyze** — Visualize delivery performance bottlenecks by region and shipping mode in Tableau
3. **Act** — Use Gen AI to generate proactive recovery actions based on model predictions

---

## 📊 Dataset

**DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS**
- Source: [Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
- 180,519 orders | 53 features
- Date range: January 2015 — January 2018

> Download the dataset from Kaggle and place the CSV file inside the `/data` folder.

---

## 🗂️ Project Structure

```
Final-Project/
│
├── data/                          # Data files (not tracked by git)
│   ├── DataCoSupplyChainDataset.csv
│   ├── df_working_clean.csv
│   ├── holdout_raw.csv
│   ├── tableau_data.csv
│   ├── test_predictions.csv
│   └── holdout_predictions.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Data prep, cleaning, encoding
│   ├── 02_eda.ipynb               # Exploratory data analysis
│   ├── 03_modeling.ipynb          # ML models and evaluation
│   └── 04_genai.ipynb             # Gen AI component and holdout evaluation
│
├── models/                        # Saved models
│   ├── tuned_xgboost.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
│
├── app/
│   └── app.py                     # Streamlit application
│
├── docs/                          # Presentation and documentation
├── .env                           # API keys (not tracked by git)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Final-Project.git
cd Final-Project

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_openai_api_key
```

---

## 🚀 Usage

**Run the Streamlit app locally:**
```bash
cd app
streamlit run app.py
```

**Run the notebooks in order:**
1. `01_data_exploration.ipynb` — Data preparation
2. `02_eda.ipynb` — Exploratory analysis
3. `03_modeling.ipynb` — Model training
4. `04_genai.ipynb` — Gen AI + holdout evaluation

---

## 🔬 Methodology

### Data Preparation
- Temporal holdout split: most recent 10% reserved for final evaluation
- Removed leaky features (post-shipment data)
- Removed PII and irrelevant identifiers
- One-Hot Encoding for low cardinality features
- Label Encoding for medium cardinality features
- Date feature engineering (month, day of week, quarter, hour)

### Key EDA Findings
- **First Class shipping** has a ~96% late delivery rate due to overpromising tight delivery windows
- Orders with **1 scheduled delivery day** have a ~95% late rate
- The core insight: **the problem is overpromising, not shipping speed**

### Models Trained

| Model | Accuracy | ROC-AUC | F1 (Late) |
|---|---|---|---|
| Logistic Regression | 70% | 0.7410 | 0.67 |
| Neural Network | 71% | 0.7729 | 0.69 |
| LightGBM | 72% | 0.8127 | 0.70 |
| XGBoost | 73% | 0.8279 | 0.72 |
| **Tuned XGBoost** | **91%** | **0.9618** | **0.91** |

### Holdout Evaluation (Unseen Future Data)

| Metric | Test Set | Holdout Set |
|---|---|---|
| Accuracy | 91% | 68% |
| ROC-AUC | 0.9618 | 0.7519 |
| F1 (Late) | 0.91 | 0.70 |

> The performance drop between test and holdout sets reflects **temporal concept drift** — the model was trained on 2015-2017 data and evaluated on Aug 2017-Jan 2018 data. This highlights the importance of regular model retraining in production.

### Top Features (XGBoost Importance)
1. Shipping Mode — Same Day (0.337)
2. Shipping Mode — First Class (0.287)
3. Days for Shipment Scheduled (0.200)
4. Shipping Mode — Second Class (0.036)

---

## 🤖 Gen AI Component

High-risk orders are passed to **GPT-4o-mini** with structured context including:
- Product details
- Customer segment
- Shipping mode and region
- Predicted risk probability

The LLM generates:
1. A customer apology message
2. Two operational recovery actions
3. A shipping upgrade recommendation

---

## 📈 Tableau Dashboard

The Tableau dashboard includes:
- Late Delivery Rate by Shipping Mode
- Late Delivery Rate by Country (Map)
- Order Volume Over Time by Shipping Mode
- Department/Category Performance Heatmap
- Late Delivery Rate by Scheduled Days

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Tableau |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Deep Learning | TensorFlow/Keras |
| Hyperparameter Tuning | Optuna |
| Gen AI | OpenAI GPT-4o-mini, LangChain |
| App | Streamlit |
| Version Control | Git, GitHub |

---

## 📁 Requirements

See [requirements.txt](requirements.txt)

---

## 👤 Author

**Fernando Gonzalez Navarro** — Data Science & Machine Learning Bootcamp Final Project