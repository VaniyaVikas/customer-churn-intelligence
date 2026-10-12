
import joblib
import pandas as pd

from src.config import MODEL_PATH
from src.data_loader import load_data
from src.preprocessing import clean_data, split_features_target


def get_feature_importance():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Train the model first: python -m src.train")

    model = joblib.load(MODEL_PATH)
    df = clean_data(load_data())
    X, _ = split_features_target(df)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = abs(classifier.coef_[0])
    else:
        raise ValueError("This classifier does not expose feature importance.")

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": values,
    })

    importance = importance.sort_values(
        "importance", ascending=False
    ).head(20)

    print(importance.to_string(index=False))
    return importance


if __name__ == "__main__":
    get_feature_importance()
