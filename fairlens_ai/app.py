"""
FairLens AI – Transparent Intelligence
Production-ready Streamlit application.
"""

from __future__ import annotations
import textwrap, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="FairLens AI – Transparent Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#F0F4FF;}

/* Sidebar */
[data-testid="stSidebar"]{background:linear-gradient(175deg,#0f1535 0%,#1e2a6e 60%,#2d3fb8 100%);}
[data-testid="stSidebar"] *{color:#dde3ff !important;}
[data-testid="stSidebar"] h2{color:#ffffff !important;font-size:1.1rem;}
[data-testid="stSidebar"] hr{border-color:#3b4fd8 !important;}

/* Hero */
.hero{background:linear-gradient(135deg,#0f1535 0%,#2541b2 50%,#6c63ff 100%);
  border-radius:20px;padding:2.5rem 3rem;color:#fff;
  box-shadow:0 12px 40px rgba(37,65,178,.35);margin-bottom:2rem;}
.hero h1{margin:0;font-size:2.6rem;font-weight:800;letter-spacing:-1px;}
.hero p{margin:.5rem 0 0;font-size:1.05rem;opacity:.85;}

/* Metric cards */
.mcard{background:#fff;border-radius:16px;padding:1.4rem 1.5rem;
  box-shadow:0 2px 20px rgba(0,0,0,.07);text-align:center;border-top:4px solid;}
.mcard .lbl{font-size:.72rem;font-weight:700;text-transform:uppercase;
  letter-spacing:1px;color:#6b7280;margin-bottom:.3rem;}
.mcard .val{font-size:2rem;font-weight:800;}

/* Section header */
.sh{font-size:1.18rem;font-weight:700;color:#0f1535;
  border-left:4px solid #2541b2;padding-left:12px;margin:1.8rem 0 .8rem;}

/* Verdict */
.vbox{border-radius:14px;padding:1.1rem 1.5rem;font-size:1.06rem;
  font-weight:500;margin:.6rem 0 1.2rem;}

/* Step pill */
.pill{display:inline-block;background:#2541b2;color:#fff;
  border-radius:50px;padding:3px 13px;font-size:.72rem;
  font-weight:700;margin-bottom:5px;letter-spacing:.6px;}

/* Button */
.stButton>button{background:linear-gradient(135deg,#2541b2,#6c63ff);
  color:#fff;border:none;border-radius:12px;padding:.55rem 1.6rem;
  font-weight:700;font-size:1rem;width:100%;
  transition:transform .15s,box-shadow .15s;}
.stButton>button:hover{transform:translateY(-2px);
  box-shadow:0 8px 24px rgba(37,65,178,.4);}
footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── utils imports ──────────────────────────────────────────────────────────
from utils.model import train_model, TrainingResult
from utils.bias import compute_bias, BiasReport
from utils.explain import compute_shap, ExplainResult

# ── helpers ────────────────────────────────────────────────────────────────
def mcard(label, value, color="#2541b2"):
    return f'<div class="mcard" style="border-color:{color}"><div class="lbl">{label}</div><div class="val" style="color:{color}">{value}</div></div>'

def sh(title):
    st.markdown(f'<div class="sh">{title}</div>', unsafe_allow_html=True)

def bias_color(s):
    return "#22c55e" if s<5 else "#eab308" if s<15 else "#f97316" if s<30 else "#ef4444"

# ── HERO ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔍 FairLens AI</h1>
  <p><strong>Transparent Intelligence</strong> — Upload a dataset · Train a model · Detect bias · Explain every decision</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown('<span class="pill">STEP 1</span>', unsafe_allow_html=True)
    st.markdown("**Upload or use sample data**")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.checkbox("📂 Use built-in sample dataset", value=False)

    st.markdown("---")
    target_col = sensitive_col = None
    run_analysis = False
    df_raw = None

    if uploaded_file or use_sample:
        try:
            df_raw = pd.read_csv("sample_data.csv") if use_sample else pd.read_csv(uploaded_file)
            cols = df_raw.columns.tolist()

            st.markdown('<span class="pill">STEP 2</span>', unsafe_allow_html=True)
            st.markdown("**Select Columns**")
            target_col = st.selectbox("🎯 Target Column", cols, index=len(cols)-1)
            rem = [c for c in cols if c != target_col]
            def_sens = next((i for i,c in enumerate(rem) if c.lower() in ("gender","sex","race","ethnicity","age")), 0)
            sensitive_col = st.selectbox("🛡️ Sensitive Feature", rem, index=def_sens)

            st.markdown("---")
            st.markdown('<span class="pill">STEP 3</span>', unsafe_allow_html=True)
            run_analysis = st.button("🚀 Run FairLens Analysis")
        except Exception as e:
            st.error(f"Could not read file: {e}")
    else:
        st.info("⬆️ Upload a CSV or use the sample dataset.")

    st.markdown("---")
    st.markdown("<small>Built with Streamlit · Fairlearn · SHAP · Scikit-learn</small>", unsafe_allow_html=True)

# ── WELCOME STATE ──────────────────────────────────────────────────────────
if df_raw is None:
    c1,c2,c3 = st.columns(3)
    with c1: st.markdown(mcard("Step 1","Upload CSV","#2541b2"), unsafe_allow_html=True)
    with c2: st.markdown(mcard("Step 2","Pick Columns","#6c63ff"), unsafe_allow_html=True)
    with c3: st.markdown(mcard("Step 3","Run Analysis","#22c55e"), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📖 How to use FairLens AI", expanded=True):
        st.markdown("""
**FairLens AI** audits any ML model for hidden bias and explains what drives each prediction.

| Feature | Description |
|---|---|
| 📂 **CSV Upload** | Any tabular dataset |
| 🤖 **Auto-training** | Logistic Regression in seconds |
| ⚖️ **Bias Detection** | Demographic Parity & Equalized Odds |
| 🔍 **SHAP Explainability** | Feature importance plots |
| 💬 **Plain-English verdict** | No jargon, just clear insights |

**Quick start:** Check *"Use built-in sample dataset"* → select columns → click **Run FairLens Analysis**.

> The sample dataset is a synthetic **loan-approval** scenario with `gender` as the sensitive attribute.
""")
    st.stop()

# ── DATA PREVIEW ──────────────────────────────────────────────────────────
sh("📋 Dataset Preview")
cl, cr = st.columns([3,1])
with cl:
    st.dataframe(df_raw.head(10), use_container_width=True)
with cr:
    st.markdown(mcard("Rows", f"{len(df_raw):,}", "#2541b2"), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(mcard("Columns", f"{len(df_raw.columns)}", "#6c63ff"), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(mcard("Missing", f"{df_raw.isnull().sum().sum()}", "#eab308"), unsafe_allow_html=True)

# distribution chart
sh(f"📊 Distribution — {sensitive_col}")
vc = df_raw[sensitive_col].value_counts()
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,3.5))
colors = ["#2541b2","#6c63ff","#a78bfa","#c4b5fd"]
ax1.bar(vc.index.astype(str), vc.values, color=colors[:len(vc)])
ax1.set_title(f"{sensitive_col} — Count", fontweight="bold"); ax1.spines[["top","right"]].set_visible(False)
ax2.pie(vc.values, labels=vc.index.astype(str), autopct="%1.1f%%",
        colors=colors[:len(vc)], startangle=140, wedgeprops={"edgecolor":"white","linewidth":2})
ax2.set_title(f"{sensitive_col} — Share", fontweight="bold")
for a in (ax1,ax2): a.set_facecolor("#F0F4FF")
fig.patch.set_facecolor("#F0F4FF"); plt.tight_layout()
st.pyplot(fig); plt.close(fig)

if not run_analysis:
    st.info("👈 Click **Run FairLens Analysis** in the sidebar when ready.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════
with st.spinner("🤖 Training Logistic Regression…"):
    try:
        result: TrainingResult = train_model(df_raw, target_col, sensitive_col)
    except Exception as e:
        st.error(f"Training failed: {e}"); st.stop()

# ── BIAS ──────────────────────────────────────────────────────────────────
with st.spinner("⚖️ Running fairness analysis…"):
    try:
        # Pull sensitive values for test rows using their original df indices
        sens_test = df_raw[sensitive_col].iloc[result.X_test.index].reset_index(drop=True)
        if sensitive_col in result.label_encoders:
            sens_test = result.label_encoders[sensitive_col].transform(sens_test.astype(str))
        bias_report: BiasReport = compute_bias(
            result.y_test.reset_index(drop=True),
            result.y_pred,
            pd.Series(sens_test, name=sensitive_col)
        )
    except Exception as e:
        st.error(f"Bias analysis failed: {e}"); st.stop()

# ── SHAP ──────────────────────────────────────────────────────────────────
with st.spinner("🔍 Computing SHAP explanations…"):
    try:
        explain: ExplainResult = compute_shap(result.model, result.X_train, result.X_test)
    except Exception as e:
        st.error(f"SHAP failed: {e}"); st.stop()

# ══════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
bc = bias_color(bias_report.bias_score)
auc_str = f"{result.roc_auc:.3f}" if not np.isnan(result.roc_auc) else "N/A"

sh("🏆 Model & Fairness Summary")
c1,c2,c3,c4 = st.columns(4)
with c1: st.markdown(mcard("Accuracy", f"{result.accuracy:.1%}", "#2541b2"), unsafe_allow_html=True)
with c2: st.markdown(mcard("ROC-AUC", auc_str, "#6c63ff"), unsafe_allow_html=True)
with c3: st.markdown(mcard("Bias Score", f"{bias_report.bias_score:.1f}/100", bc), unsafe_allow_html=True)
with c4: st.markdown(mcard("DP Δ", f"{bias_report.demographic_parity_diff:+.4f}", "#f97316"), unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f'<div class="vbox" style="background:{bc}18;border:1.5px solid {bc}">{bias_report.verdict}</div>', unsafe_allow_html=True)

# ── Fairness metrics table ─────────────────────────────────────────────────
sh("⚖️ Fairness Metrics")
st.dataframe(pd.DataFrame({
    "Metric": ["Demographic Parity Difference","Demographic Parity Ratio","Equalized Odds Difference","Equalized Odds Ratio"],
    "Value":  [bias_report.demographic_parity_diff, bias_report.demographic_parity_ratio,
               bias_report.equalized_odds_diff, bias_report.equalized_odds_ratio],
    "Ideal":  ["0.00","1.00","0.00","1.00"],
    "Interpretation": [
        "Selection-rate gap across groups (0 = perfectly fair)",
        "Selection-rate ratio across groups (1 = perfectly fair)",
        "TPR/FPR gap across groups (0 = perfectly fair)",
        "TPR/FPR ratio across groups (1 = perfectly fair)",
    ]
}), use_container_width=True, hide_index=True)

# ── Per-group metrics ──────────────────────────────────────────────────────
sh(f"📊 Per-Group Metrics — {sensitive_col}")
st.dataframe(bias_report.group_metrics, use_container_width=True, hide_index=True)

grp = bias_report.group_metrics
grp_labels = grp.iloc[:,0].astype(str)
clrs = ["#2541b2","#6c63ff","#a78bfa","#f97316"]

sh("📈 Accuracy by Group")
fig2, ax2 = plt.subplots(figsize=(8,3.5))
ax2.bar(grp_labels, grp["Accuracy"], color=clrs[:len(grp)])
ax2.axhline(result.accuracy, color="#ef4444", linestyle="--", lw=1.5, label=f"Overall ({result.accuracy:.1%})")
ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.spines[["top","right"]].set_visible(False)
ax2.set_title(f"Accuracy per {sensitive_col} Group", fontweight="bold")
ax2.set_facecolor("#F0F4FF"); fig2.patch.set_facecolor("#F0F4FF"); plt.tight_layout()
st.pyplot(fig2); plt.close(fig2)

if "Selection Rate" in grp.columns:
    sh("🎯 Selection Rate by Group")
    fig3, ax3 = plt.subplots(figsize=(8,3.5))
    ax3.bar(grp_labels, grp["Selection Rate"], color=["#f97316","#eab308","#22c55e","#14b8a6"][:len(grp)])
    ax3.set_ylabel("Selection Rate"); ax3.spines[["top","right"]].set_visible(False)
    ax3.set_title(f"Selection Rate per {sensitive_col} Group", fontweight="bold")
    ax3.set_facecolor("#F0F4FF"); fig3.patch.set_facecolor("#F0F4FF"); plt.tight_layout()
    st.pyplot(fig3); plt.close(fig3)

# ── Confusion matrix ──────────────────────────────────────────────────────
sh("🧮 Confusion Matrix")
cm = result.conf_matrix
fig4, ax4 = plt.subplots(figsize=(4,3.5))
ax4.imshow(cm, cmap="Blues")
ax4.set_xticks([0,1]); ax4.set_yticks([0,1])
ax4.set_xticklabels(["Pred 0","Pred 1"]); ax4.set_yticklabels(["Actual 0","Actual 1"])
for i in range(2):
    for j in range(2):
        ax4.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=18,fontweight="bold",
                 color="white" if cm[i,j]>cm.max()/2 else "#0f1535")
ax4.set_title("Confusion Matrix", fontweight="bold")
fig4.patch.set_facecolor("#F0F4FF"); plt.tight_layout()
col_cm, _ = st.columns([1,2])
with col_cm: st.pyplot(fig4)
plt.close(fig4)

# ── SHAP ──────────────────────────────────────────────────────────────────
sh("🔍 SHAP Feature Importance")
st.pyplot(explain.bar_fig); plt.close(explain.bar_fig)

sh("🌊 SHAP Summary (Bee Swarm)")
st.pyplot(explain.summary_fig); plt.close(explain.summary_fig)

sh("📋 Top Features — Mean |SHAP|")
fi = explain.feature_importance.head(10).copy()
fi.columns = ["Feature","Mean |SHAP|"]
fi["Mean |SHAP|"] = fi["Mean |SHAP|"].round(4)
st.dataframe(fi, use_container_width=True, hide_index=True)

# ── Plain-language explanation ─────────────────────────────────────────────
sh("💬 Plain-Language Explanation")
top_f = explain.feature_importance.iloc[0]["feature"]
top_s = explain.feature_importance.iloc[0]["mean_abs_shap"]
st.markdown(textwrap.dedent(f"""
**What did the model learn?**
Trained on **{len(df_raw):,} records**, evaluated on **{len(result.y_test):,} test samples**.
- **Accuracy:** {result.accuracy:.1%} &nbsp;|&nbsp; **ROC-AUC:** {auc_str}

**What drives the predictions?**
The most influential feature is **`{top_f}`** (mean |SHAP| = {top_s:.4f}).

**Is the model fair?**
- Demographic Parity Δ = **{bias_report.demographic_parity_diff:+.4f}** (ideal: 0)
- Equalized Odds Δ = **{bias_report.equalized_odds_diff:+.4f}** (ideal: 0)
- Bias Score = **{bias_report.bias_score:.1f} / 100**

{bias_report.verdict}

**Next steps:** {"✅ Model appears fair — keep monitoring on live data." if bias_report.bias_score < 15 else "⚠️ Consider reweighing samples or applying Fairlearn's ExponentiatedGradient to reduce bias."}
"""))

with st.expander("📄 Full Classification Report"):
    st.code(result.report, language="text")

# ── Download ───────────────────────────────────────────────────────────────
sh("⬇️ Export Results")
metrics_df = pd.DataFrame({
    "Metric": ["Demographic Parity Difference","Demographic Parity Ratio","Equalized Odds Difference","Equalized Odds Ratio"],
    "Value":  [bias_report.demographic_parity_diff, bias_report.demographic_parity_ratio,
               bias_report.equalized_odds_diff, bias_report.equalized_odds_ratio],
    "Ideal":  ["0.00","1.00","0.00","1.00"],
})
st.download_button("📥 Download Fairness Metrics CSV", metrics_df.to_csv(index=False).encode(),
                   "fairlens_metrics.csv", "text/csv")

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border:none;border-top:1.5px solid #dde3ff;margin:2rem 0">
<p style="text-align:center;color:#6b7280;font-size:.85rem">
  🔍 <strong>FairLens AI</strong> — Transparent Intelligence &nbsp;|&nbsp;
  Streamlit · Fairlearn · SHAP · Scikit-learn
</p>
""", unsafe_allow_html=True)
