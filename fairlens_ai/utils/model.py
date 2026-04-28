"""
model.py – Model training utilities for FairLens AI
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingResult:
    model: Any
    scaler: StandardScaler
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    y_prob: np.ndarray
    feature_names: list[str]
    accuracy: float
    roc_auc: float
    report: str
    conf_matrix: np.ndarray
    label_encoders: dict = field(default_factory=dict)


def encode_dataframe(
    df: pd.DataFrame, target_col: str, sensitive_col: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    Encode categorical columns and return clean feature matrix and targets.
    Returns (X, y, sensitive_series, label_encoders).
    """
    df = df.copy()
    label_encoders: dict = {}

    # Encode target
    if df[target_col].dtype == object:
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col].astype(str))
        label_encoders[target_col] = le

    # Keep sensitive column as-is (needed for fairlearn), but encode if str
    if df[sensitive_col].dtype == object:
        le = LabelEncoder()
        df[sensitive_col] = le.fit_transform(df[sensitive_col].astype(str))
        label_encoders[sensitive_col] = le

    # Encode remaining object columns
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    y = df[target_col]
    X = df.drop(columns=[target_col])
    sensitive_series = df[sensitive_col]

    return X, y, sensitive_series, label_encoders


def train_model(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
    test_size: float = 0.25,
    random_state: int = 42,
) -> TrainingResult:
    """
    Encode data, scale features, train Logistic Regression, and return results.
    """
    if df.isnull().values.any():
        df = df.dropna()

    X, y, sensitive_series, label_encoders = encode_dataframe(
        df, target_col, sensitive_col
    )

    # Ensure binary target (pick top 2 classes if multiclass)
    if y.nunique() > 2:
        top2 = y.value_counts().index[:2].tolist()
        mask = y.isin(top2)
        X, y, sensitive_series = X[mask], y[mask], sensitive_series[mask]
        mapping = {top2[0]: 0, top2[1]: 1}
        y = y.map(mapping)

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive_series, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        roc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc = float("nan")

    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return TrainingResult(
        model=model,
        scaler=scaler,
        X_train=X_train_s,
        X_test=X_test_s,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        feature_names=list(X.columns),
        accuracy=acc,
        roc_auc=roc,
        report=report,
        conf_matrix=cm,
        label_encoders=label_encoders,
    )
