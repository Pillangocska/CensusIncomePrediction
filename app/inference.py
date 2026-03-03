from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "_models"

WORKCLASS_CATS = [
    "Local-gov", "Private", "Self-emp-inc", "Self-emp-not-inc",
    "State-gov", "Unknown",
]

OCCUPATION_CATS = [
    "Armed-Forces", "Craft-repair", "Exec-managerial", "Farming-fishing",
    "Handlers-cleaners", "Machine-op-inspct", "Other-service",
    "Priv-house-serv", "Prof-specialty", "Protective-serv", "Sales",
    "Tech-support", "Transport-moving", "Unknown",
]

MARRIED_VALUES = {"Married-civ-spouse", "Married-AF-spouse"}


def load_models() -> tuple:
    """Load models and feature columns. Returns (xgb_default, xgb_balanced, feature_columns).

    Raises FileNotFoundError with details if any file is missing.
    """
    required_files = {
        "xgb_default.joblib": MODEL_DIR / "xgb_default.joblib",
        "xgb_balanced.joblib": MODEL_DIR / "xgb_balanced.joblib",
        "feature_columns.joblib": MODEL_DIR / "feature_columns.joblib",
    }

    missing = [name for name, path in required_files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model files in {MODEL_DIR}: {', '.join(missing)}"
        )

    xgb_default = joblib.load(required_files["xgb_default.joblib"])
    xgb_balanced = joblib.load(required_files["xgb_balanced.joblib"])
    feature_columns = joblib.load(required_files["feature_columns.joblib"])
    return xgb_default, xgb_balanced, feature_columns


def preprocess_person(raw_input: dict, feature_columns: list) -> pd.DataFrame:
    """Convert a raw person dictionary into a model-ready DataFrame."""
    row = {}

    row["age"] = raw_input["age"]
    row["education_num"] = raw_input["education_num"]
    row["hours_per_week"] = raw_input["hours_per_week"]

    row["capital_gain"] = np.log1p(raw_input["capital_gain"])
    row["capital_loss"] = np.log1p(raw_input["capital_loss"])

    row["sex"] = 1 if raw_input["sex"] == "Male" else 0
    row["marital_status"] = 1 if raw_input["marital_status"] in MARRIED_VALUES else 0
    row["native_country"] = 1 if raw_input["native_country"] == "United-States" else 0

    for cat in WORKCLASS_CATS:
        row[f"workclass_{cat}"] = 1 if raw_input.get("workclass", "Unknown") == cat else 0

    for cat in OCCUPATION_CATS:
        row[f"occupation_{cat}"] = 1 if raw_input.get("occupation", "Unknown") == cat else 0

    return pd.DataFrame([row])[feature_columns]


def predict_income(raw_input: dict, model, feature_columns: list) -> tuple[str, float]:
    """Predict income class for a person. Returns (prediction, probability)."""
    X = preprocess_person(raw_input, feature_columns)
    proba = model.predict_proba(X)[0][1]
    prediction = ">50K" if proba >= 0.5 else "<=50K"
    return prediction, proba
