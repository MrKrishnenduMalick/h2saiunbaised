"""
explain.py – SHAP-based explainability utilities for FairLens AI
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression


@dataclass
class ExplainResult:
    shap_values: np.ndarray
    base_value: float
    feature_importance: pd.DataFrame   # columns: feature, mean_abs_shap
    summary_fig: plt.Figure
    bar_fig: plt.Figure


def compute_shap(
    model: LogisticRegression,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    max_display: int = 15,
) -> ExplainResult:
    """
    Compute SHAP values for a fitted LogisticRegression and return figures.
    """
    # Use LinearExplainer – fast and exact for linear models
    explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
    shap_vals = explainer.shap_values(X_test)

    # Feature importance: mean |SHAP|
    mean_abs = np.abs(shap_vals).mean(axis=0)
    fi_df = pd.DataFrame(
        {"feature": X_test.columns.tolist(), "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # ── Summary (bee-swarm style) ──────────────────────────────────────────
    fig_summary, ax_summary = plt.subplots(figsize=(9, 5))
    shap.summary_plot(
        shap_vals,
        X_test,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    fig_summary = plt.gcf()
    fig_summary.patch.set_facecolor("#F8F9FB")
    plt.tight_layout()

    # ── Bar chart (mean |SHAP|) ────────────────────────────────────────────
    top = fi_df.head(max_display)
    fig_bar, ax_bar = plt.subplots(figsize=(9, max(4, len(top) * 0.42)))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(top)))
    ax_bar.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color=colors[::-1])
    ax_bar.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax_bar.set_title("Feature Importance (SHAP)", fontsize=13, fontweight="bold")
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.set_facecolor("#F8F9FB")
    fig_bar.patch.set_facecolor("#F8F9FB")
    plt.tight_layout()

    return ExplainResult(
        shap_values=shap_vals,
        base_value=explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[1],
        feature_importance=fi_df,
        summary_fig=fig_summary,
        bar_fig=fig_bar,
    )
