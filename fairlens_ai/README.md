# 🔍 FairLens AI – Transparent Intelligence

> **AI bias detection and model explainability — powered by Streamlit, Fairlearn & SHAP**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📂 CSV Upload | Upload any tabular dataset or use the built-in sample |
| 🤖 Auto-Training | Logistic Regression trained in seconds |
| ⚖️ Bias Detection | Demographic Parity & Equalized Odds via Fairlearn |
| 🔍 SHAP Explainability | Feature importance + bee-swarm summary plots |
| 📊 Rich Dashboard | Colour-coded KPI cards, group charts, confusion matrix |
| 💬 Plain-English | Auto-generated, jargon-free verdict |
| ⬇️ Export | Download fairness metrics as CSV |

---

## 🗂️ Project Structure

```
fairlens_ai/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
├── runtime.txt         ← Python 3.11
├── sample_data.csv     ← Built-in loan-approval demo dataset
└── utils/
    ├── __init__.py
    ├── model.py        ← Logistic Regression training pipeline
    ├── bias.py         ← Fairlearn bias/fairness metrics
    └── explain.py      ← SHAP explanations & visualisations
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.11+
- pip

### Install & Run

```bash
# 1. Clone / enter the project folder
cd fairlens_ai

# 2. Create virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**.

---

## ☁️ Deploy to Streamlit Community Cloud (Step-by-Step)

### Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit – FairLens AI"
git branch -M main
git remote add origin https://github.com/<your-username>/fairlens-ai.git
git push -u origin main
```

### Step 2 — Create a Streamlit Cloud account

1. Go to **https://share.streamlit.io**
2. Sign in with GitHub.

### Step 3 — Deploy the app

1. Click **"New app"** (top-right).
2. Fill in:
   - **Repository:** `<your-username>/fairlens-ai`
   - **Branch:** `main`
   - **Main file path:** `app.py`
3. Click **"Deploy"**.

Streamlit Cloud will:
- Read `runtime.txt` → use Python 3.11
- Read `requirements.txt` → install all packages
- Start the app automatically

### Step 4 — Access your live URL

Your app will be live at:

```
https://<your-app-name>.streamlit.app
```

> **Tip:** You can customise the subdomain in **App settings → General → App URL**.

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: fairlearn` | Ensure `fairlearn>=0.10.0` is in `requirements.txt` |
| `SHAP import error` | Pin `shap>=0.45.0` in `requirements.txt` |
| App crashes on upload | Ensure CSV has no completely empty columns; drop them first |
| `runtime.txt` ignored | Confirm the file contains exactly `python-3.11` with no extra spaces |
| Slow cold start | Normal on free tier — first load may take ~60 seconds |

---

## 📦 Dependencies

```
streamlit>=1.32.0
scikit-learn>=1.4.0
fairlearn>=0.10.0
shap>=0.45.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
```

---

## 📄 License

MIT © 2024 FairLens AI
