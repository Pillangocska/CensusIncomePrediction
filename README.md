# Census Income Prediction

End-to-end ML pipeline for binary income classification on the [UCI Adult Census dataset](https://archive.ics.uci.edu/dataset/2/adult). Predicts whether a person earns more than $50K/year based on demographic and employment attributes.

## Project Structure

```
CensusIncomePrediction/
├── README.md
├── pyproject.toml                  # Project config & dependencies
├── main.py                         # Entry point (stub)
├── _data/
│   ├── adult.csv                   # Original dataset (32,561 rows, 15 columns)
│   ├── X_train.csv                 # Preprocessed training features (26,048 x 28)
│   ├── X_test.csv                  # Preprocessed test features (6,513 x 28)
│   ├── y_train.csv                 # Training labels
│   └── y_test.csv                  # Test labels
├── app/
│   ├── __init__.py
│   ├── inference.py                # Inference module (preprocessing + prediction)
│   └── streamlit_app.py            # Streamlit web UI
├── _models/
│   ├── xgb_default.joblib          # Default XGBoost model
│   ├── xgb_balanced.joblib         # Balanced XGBoost model (recall-focused)
│   └── feature_columns.joblib      # Feature ordering for inference
└── notebook/
    ├── 0_Profiling_and_EDA.ipynb
    ├── 1_Preprocess_and_Feature_Engineering.ipynb
    ├── 2_Modeling_and_Evaluation.ipynb
    ├── 3_Inference.ipynb
    └── mlruns/                     # MLflow experiment tracking data
```

## Pipeline Overview

The project follows a four-stage notebook pipeline, designed to be run in order:

### 1. Profiling & EDA ([0_Profiling_and_EDA.ipynb](notebook/0_Profiling_and_EDA.ipynb))

Explores the raw dataset to understand distributions, correlations, and class imbalance.

**Key findings:**
- Target is imbalanced: 76% earn <=50K, 24% earn >50K
- Strongest predictors: `marital_status`, `occupation`, `education_num`, `age`
- Missing values in `workclass` (1,836) and `occupation` (1,843), largely overlapping — likely not-in-labor-force individuals
- `education` and `education_num` have a perfect 1-to-1 mapping (redundant)
- `capital_gain` / `capital_loss` are extremely sparse (92%/95% zeros) but carry high signal

### 2. Preprocessing & Feature Engineering ([1_Preprocess_and_Feature_Engineering.ipynb](notebook/1_Preprocess_and_Feature_Engineering.ipynb))

Transforms raw data into model-ready features (15 raw columns to 28 engineered features).

**Steps:**
- **Dropped columns:** `fnlwgt` (census weight, not predictive), `education` (redundant with `education_num`), `race` (ethical concerns, low signal), `relationship` (redundant with `marital_status` + `sex`)
- **Missing values:** Replaced `?` with `Unknown` for `workclass` and `occupation`
- **Binary encoding:** `marital_status` (married vs. not), `native_country` (US vs. non-US), `sex` (male/female), `income` (>50K = 1)
- **Log transform:** `capital_gain` and `capital_loss` via `log1p()` to handle extreme skew
- **One-hot encoding:** `workclass` (6 features) and `occupation` (14 features)
- **Train/test split:** 80/20, stratified by target class, random seed 42

### 3. Modeling & Evaluation ([2_Modeling_and_Evaluation.ipynb](notebook/2_Modeling_and_Evaluation.ipynb))

Trains and compares multiple models with MLflow experiment tracking.

**Models evaluated:**

| Model | Accuracy | ROC AUC | >50K Precision | >50K Recall | >50K F1 |
|-------|----------|---------|----------------|-------------|---------|
| Baseline (majority class) | 0.759 | — | — | — | — |
| Random Forest (200 trees) | 0.848 | 0.891 | 0.711 | 0.620 | 0.662 |
| **XGBoost (default)** | **0.873** | **0.925** | **0.784** | **0.652** | **0.712** |
| XGBoost (balanced) | 0.831 | 0.924 | 0.608 | 0.847 | 0.708 |
| XGBoost (Optuna, 100 trials) | 0.871 | 0.924 | 0.786 | 0.640 | 0.705 |
| Extra Trees (default) | 0.834 | 0.848 | 0.671 | 0.605 | 0.636 |
| Extra Trees (Optuna, 100 trials) | 0.865 | 0.914 | 0.779 | 0.614 | 0.687 |

**Best overall:** XGBoost (default) — best accuracy (0.873) and ROC-AUC (0.925) with no tuning needed.

**Balanced variant:** XGBoost with `scale_pos_weight=3.15` achieves the highest recall (0.847), useful when catching all high earners matters more than precision.

**Hyperparameter tuning:** Optuna (100 trials, 5-fold CV) was run for both XGBoost and Extra Trees. Tuned XGBoost performed comparably to the default, suggesting the defaults are already well-suited to this dataset.

**Additional analysis includes:** ROC curves, precision-recall curves, confusion matrices, cumulative gains charts, feature importance plots, and threshold sensitivity analysis.

### 4. Inference ([3_Inference.ipynb](notebook/3_Inference.ipynb))

Production-ready prediction pipeline with two exported models.

**Usage:**
```python
raw_input = {
    "age": 45,
    "education_num": 13,           # Bachelors
    "marital_status": "Married-civ-spouse",
    "sex": "Male",
    "workclass": "Private",
    "occupation": "Exec-managerial",
    "capital_gain": 5000,
    "capital_loss": 0,
    "hours_per_week": 50,
    "native_country": "United-States"
}

prediction, probability = predict_income(raw_input, model_type="default")
# prediction: '>50K', probability: 1.0
```

**Available models:**
- `default` — Precision-focused (fewer false positives)
- `balanced` — Recall-focused (catches more high earners)

**Input fields:**

| Field | Type | Range / Values |
|-------|------|----------------|
| `age` | int | 17–90 |
| `education_num` | int | 1–16 (1=Preschool, 16=Doctorate) |
| `marital_status` | str | Married-civ-spouse, Never-married, Divorced, Separated, Widowed, Married-spouse-absent, Married-AF-spouse |
| `sex` | str | Male, Female |
| `workclass` | str | Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Unknown |
| `occupation` | str | Exec-managerial, Prof-specialty, Craft-repair, Sales, Tech-support, etc. (14 categories) |
| `capital_gain` | int | 0+ |
| `capital_loss` | int | 0+ |
| `hours_per_week` | int | 1–99 |
| `native_country` | str | Country name (binarized to US vs. non-US) |

## Streamlit Web App

The `app/` folder contains an interactive web interface for making predictions.

**Features:**
- Sidebar with input controls for all person attributes (sliders, dropdowns)
- Three pre-built example profiles: High earner, Lower earner, Borderline case
- Model selection: Default, Balanced, or Compare Both side-by-side
- Education level displayed with human-readable labels (e.g. "13 - Bachelors")

**Running the app:**
```bash
cd app
streamlit run streamlit_app.py
```

**Architecture:** The app uses [inference.py](app/inference.py), a standalone module that handles model loading, preprocessing, and prediction — making it reusable outside the Streamlit context.

## MLflow Tracking

All model training runs are tracked with MLflow, stored locally in `notebook/mlruns/`.

**Tracked per run:**
- Hyperparameters and model configuration
- Metrics: accuracy, ROC-AUC, precision, recall, F1 (for the >50K class)
- Artifacts: serialized models, environment specs, evaluation plots

To browse experiments:
```bash
cd notebook
mlflow ui
```

## Setup

### Requirements

- Python >=3.12, <3.14

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd CensusIncomePrediction

# Install dependencies (using uv, pip, or any PEP 621-compatible tool)
pip install -e .
```

### Key Dependencies

- `xgboost` — Gradient boosting models
- `scikit-learn` — Random Forest, Extra Trees, preprocessing utilities
- `optuna` — Hyperparameter optimization
- `mlflow` — Experiment tracking
- `streamlit` — Web UI for interactive predictions
- `pandas`, `numpy` — Data manipulation
- `matplotlib`, `seaborn` — Visualization
- `joblib` — Model serialization

## Dataset

The [Adult Census Income dataset](https://archive.ics.uci.edu/dataset/2/adult) from the UCI Machine Learning Repository. Extracted from the 1994 US Census database.

- **Samples:** 32,561
- **Features:** 14 (6 numeric, 8 categorical) + 1 target
- **Task:** Binary classification (income >50K vs. <=50K)
- **Class distribution:** 75.9% <=50K, 24.1% >50K

## Reproducibility

- Random seed 42 used across all splits, models, and Optuna studies
- Stratified train/test split preserves class balance
- Feature column ordering saved to `_models/feature_columns.joblib`
- MLflow logs full environment specs (conda & pip) per run
