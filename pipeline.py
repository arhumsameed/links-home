"""Data loading, encoding and model training for the Links-Home cold chain app.

"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

# Relative to this file, so it works on your laptop, in CI and on Streamlit Cloud.
DATA_PATH = Path(__file__).parent / "iot_cold_chain_logistics.csv"
TARGET_COL = "Delivery_Status"

FEATURE_COLS = [
    "Temperature_C",
    "Temperature_Variation",
    "Vehicle_Age_Years",
    "Cooling_Load_Percent",
    "Route_Distance_km",
    "Traffic_Level",
    "Road_Condition",
    "Transit_Delay_hr",
]

# One mapping, used for training AND for the dispatch form, so they cannot drift apart.
# If Traffic_Level / Road_Condition are already numeric in the CSV, they are used as-is
# and are assumed to follow the same 1..4 / 1..3 coding.
TRAFFIC_MAP = {"low": 1, "medium": 2, "high": 3, "severe": 4, "none": 1}
TRAFFIC_LABELS = {1: "Low", 2: "Medium", 3: "High", 4: "Severe"}
ROAD_MAP = {"good": 1, "moderate": 2, "poor": 3}
ROAD_LABELS = {1: "Good (Highway)", 2: "Moderate", 3: "Poor (Rural/Damaged)"}


def encode_features(df):
    """Return a copy of df with every feature column numeric."""
    df = df.copy()
    for col in FEATURE_COLS:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            cleaned = df[col].astype(str).str.strip().str.lower()
            if col == "Traffic_Level":
                df[col] = cleaned.map(TRAFFIC_MAP).fillna(2).astype(int)
            elif col == "Road_Condition":
                df[col] = cleaned.map(ROAD_MAP).fillna(2).astype(int)
            else:
                df[col], _ = pd.factorize(df[col])
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def load_data(path=DATA_PATH):
    return encode_features(pd.read_csv(path))


def train_model(df, features=None, n_estimators=100, seed=42):
    """Train a Random Forest. Returns (model, X_test, y_test) so it can be evaluated."""
    features = list(features or FEATURE_COLS)
    X = df[features]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(X_train, y_train)
    return clf, X_test, y_test


def evaluate(clf, X_test, y_test):
    """Accuracy, macro F1 and confusion matrix on the held-out test set."""
    preds = clf.predict(X_test)
    labels = sorted(clf.classes_)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "labels": labels,
        "confusion": confusion_matrix(y_test, preds, labels=labels),
    }


def feature_importance(clf, features):
    return (
        pd.DataFrame({"Feature": list(features), "Importance": clf.feature_importances_})
        .sort_values("Importance", ascending=True)
        .reset_index(drop=True)
    )
