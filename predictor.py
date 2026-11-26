"""Loads the saved Decision Tree and predicts. No rules are written here:
the category always comes from model.predict()."""
from functools import lru_cache
import joblib
import pandas as pd

MODEL_PATH = "models/decision_tree_model.joblib"


@lru_cache(maxsize=1)
def load_bundle():
    return joblib.load(MODEL_PATH)


def predict_category(study, sleep, physical, social) -> str:
    bundle = load_bundle()
    raw = {
        "Study_Hours_Per_Day": study,
        "Sleep_Hours_Per_Day": sleep,
        "Physical_Activity_Hours_Per_Day": physical,
        "Social_Hours_Per_Day": social,
    }
    # Same feature order as training, taken from the saved bundle
    X = pd.DataFrame([[raw[f] for f in bundle["feature_names"]]], columns=bundle["feature_names"])
    return str(bundle["model"].predict(X)[0])


def global_feature_importance() -> pd.DataFrame:
    bundle = load_bundle()
    return pd.DataFrame({
        "Fitur": [f.replace("_Hours_Per_Day", "").replace("_", " ") for f in bundle["feature_names"]],
        "Kontribusi relatif": bundle["model"].feature_importances_,
    }).set_index("Fitur")
