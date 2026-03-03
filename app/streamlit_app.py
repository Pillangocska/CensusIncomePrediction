import streamlit as st

from inference import load_models, predict_income

# --- Page config ---
st.set_page_config(page_title="Census Income Prediction", layout="wide")
st.title("Census Income Prediction")
st.markdown(
    "Predict whether a person earns **>50K** or **<=50K** annually "
    "using trained XGBoost models on the UCI Adult Census dataset."
)

# --- Load models ---
try:
    xgb_default, xgb_balanced, feature_columns = load_models()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# --- Example profiles ---
EXAMPLE_PROFILES = {
    "Custom": None,
    "High earner": {
        "age": 45,
        "workclass": "Private",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "sex": "Male",
        "capital_gain": 15000,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States",
    },
    "Lower earner": {
        "age": 23,
        "workclass": "Private",
        "education_num": 9,
        "marital_status": "Never-married",
        "occupation": "Other-service",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 30,
        "native_country": "United-States",
    },
    "Borderline case": {
        "age": 38,
        "workclass": "Self-emp-not-inc",
        "education_num": 11,
        "marital_status": "Married-civ-spouse",
        "occupation": "Craft-repair",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "Other",
    },
}

EDUCATION_LABELS = {
    1: "1 - Preschool",
    2: "2 - 1st-4th",
    3: "3 - 5th-6th",
    4: "4 - 7th-8th",
    5: "5 - 9th",
    6: "6 - 10th",
    7: "7 - 11th",
    8: "8 - 12th",
    9: "9 - HS-grad",
    10: "10 - Some-college",
    11: "11 - Assoc-voc",
    12: "12 - Assoc-acdm",
    13: "13 - Bachelors",
    14: "14 - Masters",
    15: "15 - Prof-school",
    16: "16 - Doctorate",
}

MARITAL_OPTIONS = [
    "Married-civ-spouse", "Married-AF-spouse", "Never-married",
    "Divorced", "Separated", "Widowed", "Married-spouse-absent",
]

WORKCLASS_OPTIONS = [
    "Private", "Self-emp-not-inc", "Self-emp-inc",
    "Federal-gov", "Local-gov", "State-gov", "Unknown",
]

OCCUPATION_OPTIONS = [
    "Exec-managerial", "Prof-specialty", "Craft-repair", "Sales",
    "Adm-clerical", "Other-service", "Machine-op-inspct",
    "Transport-moving", "Handlers-cleaners", "Farming-fishing",
    "Tech-support", "Protective-serv", "Priv-house-serv",
    "Armed-Forces", "Unknown",
]

COUNTRY_OPTIONS = ["United-States", "Other"]

# --- Sidebar ---
st.sidebar.header("Input Parameters")

profile_name = st.sidebar.selectbox("Example Profile", list(EXAMPLE_PROFILES.keys()))
profile = EXAMPLE_PROFILES[profile_name]


def _default(key, fallback):
    """Get value from selected profile or use fallback."""
    if profile is not None:
        return profile[key]
    return fallback


def _index(options, key, fallback):
    """Get the index of a value in the options list."""
    val = _default(key, fallback)
    return options.index(val) if val in options else 0


age = st.sidebar.slider("Age", 17, 90, _default("age", 30))

edu_label = st.sidebar.selectbox(
    "Education Level",
    list(EDUCATION_LABELS.values()),
    index=_default("education_num", 9) - 1,
)
education_num = int(edu_label.split(" - ")[0])

marital_status = st.sidebar.selectbox(
    "Marital Status", MARITAL_OPTIONS,
    index=_index(MARITAL_OPTIONS, "marital_status", "Never-married"),
)

sex = st.sidebar.selectbox("Sex", ["Male", "Female"],
                           index=0 if _default("sex", "Male") == "Male" else 1)

workclass = st.sidebar.selectbox(
    "Workclass", WORKCLASS_OPTIONS,
    index=_index(WORKCLASS_OPTIONS, "workclass", "Private"),
)

occupation = st.sidebar.selectbox(
    "Occupation", OCCUPATION_OPTIONS,
    index=_index(OCCUPATION_OPTIONS, "occupation", "Other-service"),
)

capital_gain = st.sidebar.number_input("Capital Gain ($)", 0, 99999,
                                       _default("capital_gain", 0))
capital_loss = st.sidebar.number_input("Capital Loss ($)", 0, 4356,
                                       _default("capital_loss", 0))

hours_per_week = st.sidebar.slider("Hours per Week", 1, 99,
                                   _default("hours_per_week", 40))

native_country = st.sidebar.selectbox(
    "Native Country", COUNTRY_OPTIONS,
    index=_index(COUNTRY_OPTIONS, "native_country", "United-States"),
)

st.sidebar.markdown("---")
model_choice = st.sidebar.radio(
    "Model",
    ["Default", "Balanced", "Compare Both"],
    help="**Default**: higher precision (fewer false positives). "
         "**Balanced**: higher recall (catches more >50K earners).",
)

predict_clicked = st.sidebar.button("Predict", type="primary", use_container_width=True)

# --- Prediction ---
if predict_clicked:
    raw_input = {
        "age": age,
        "education_num": education_num,
        "marital_status": marital_status,
        "sex": sex,
        "workclass": workclass,
        "occupation": occupation,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": hours_per_week,
        "native_country": native_country,
    }

    st.subheader("Prediction Results")

    if model_choice == "Compare Both":
        col1, col2 = st.columns(2)

        pred_def, proba_def = predict_income(raw_input, xgb_default, feature_columns)
        pred_bal, proba_bal = predict_income(raw_input, xgb_balanced, feature_columns)

        with col1:
            st.markdown("### Default Model")
            st.metric("Predicted Class", pred_def)
            st.metric("Probability (>50K)", f"{proba_def:.3f}")
            st.caption("Higher precision - fewer false positives")

        with col2:
            st.markdown("### Balanced Model")
            st.metric("Predicted Class", pred_bal)
            st.metric("Probability (>50K)", f"{proba_bal:.3f}")
            st.caption("Higher recall - catches more >50K earners")
    else:
        model = xgb_default if model_choice == "Default" else xgb_balanced
        label = "Default" if model_choice == "Default" else "Balanced"
        pred, proba = predict_income(raw_input, model, feature_columns)

        st.markdown(f"### {label} Model")
        st.metric("Predicted Class", pred)
        st.metric("Probability (>50K)", f"{proba:.3f}")
