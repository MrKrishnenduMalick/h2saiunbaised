"""
bias.py – Fairness / bias detection utilities for FairLens AI
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
    true_positive_rate,
)
from sklearn.metrics import accuracy_score


@dataclass
class BiasReport:
    demographic_parity_diff: float
    demographic_parity_ratio: float
    equalized_odds_diff: float
    equalized_odds_ratio: float
    metric_frame: MetricFrame
    group_metrics: pd.DataFrame
    bias_score: float          # 0-100 (higher = more biased)
    verdict: str               # Human readable verdict


def _bias_score(dp_diff: float, eo_diff: float) -> float:
    """
    Composite bias score 0-100.
    0 = perfectly fair, 100 = extremely biased.
    """
    raw = (abs(dp_diff) + abs(eo_diff)) / 2.0
    score = min(raw * 100, 100.0)
    return round(score, 2)


def _verdict(score: float) -> str:
    if score < 5:
        return "✅ Very Low Bias — Model appears fair across groups."
    elif score < 15:
        return "🟡 Low Bias — Minor disparity detected; monitor carefully."
    elif score < 30:
        return "🟠 Moderate Bias — Significant disparity; mitigation recommended."
    else:
        return "🔴 High Bias — Model shows strong unfairness; action required."


def compute_bias(
    y_true: pd.Series,
    y_pred: np.ndarray,
    sensitive_series: pd.Series,
) -> BiasReport:
    """
    Compute fairness metrics using Fairlearn.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sensitive = np.array(sensitive_series)

    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
    dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)
    eo_ratio = equalized_odds_ratio(y_true, y_pred, sensitive_features=sensitive)

    mf = MetricFrame(
        metrics={
            "Accuracy": accuracy_score,
            "Selection Rate": selection_rate,
            "True Positive Rate": true_positive_rate,
            "False Positive Rate": false_positive_rate,
            "False Negative Rate": false_negative_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )

    group_metrics = mf.by_group.reset_index()
    group_metrics.columns = [str(c) for c in group_metrics.columns]

    score = _bias_score(dp_diff, eo_diff)
    verdict = _verdict(score)

    return BiasReport(
        demographic_parity_diff=round(dp_diff, 4),
        demographic_parity_ratio=round(dp_ratio, 4),
        equalized_odds_diff=round(eo_diff, 4),
        equalized_odds_ratio=round(eo_ratio, 4),
        metric_frame=mf,
        group_metrics=group_metrics,
        bias_score=score,
        verdict=verdict,
    )
