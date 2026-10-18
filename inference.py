import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from feature_generator import (
    AdaptiveSphericalRBF,
    FeatureSelector,
    PhysicsFeatureGenerator,
)

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "models/threshold.json", encoding="utf-8") as threshold_file:
    threshold_data = json.load(threshold_file)

model_sklearn_version = threshold_data.get("scikit_learn_version")
if model_sklearn_version is None:
    raise RuntimeError(
        "This model was trained with an older notebook and has no recorded "
        "scikit-learn version. Run all cells in "
        "models/predicting-tsunamis.ipynb to retrain it before starting the app."
    )
if model_sklearn_version != sklearn.__version__:
    raise RuntimeError(
        f"The model was trained with scikit-learn {model_sklearn_version}, "
        f"but this environment has {sklearn.__version__}. Install the project "
        "requirements and retrain the model using the same environment."
    )


def _load_model(path: Path):
    # Older notebook artifacts store custom transformers as __main__ classes.
    main_module = sys.modules["__main__"]
    legacy_classes = (
        PhysicsFeatureGenerator,
        AdaptiveSphericalRBF,
        FeatureSelector,
    )
    previous_classes = {
        cls.__name__: getattr(main_module, cls.__name__, None)
        for cls in legacy_classes
    }
    missing_classes = {
        cls.__name__
        for cls in legacy_classes
        if not hasattr(main_module, cls.__name__)
    }

    for cls in legacy_classes:
        setattr(main_module, cls.__name__, cls)
    try:
        return joblib.load(path)
    finally:
        for cls in legacy_classes:
            if cls.__name__ in missing_classes:
                delattr(main_module, cls.__name__)
            else:
                setattr(main_module, cls.__name__, previous_classes[cls.__name__])


model = _load_model(BASE_DIR / "models/rf_pipeline.joblib")
threshold = threshold_data["threshold"]


def predict_tsunami(input_data: dict):
    # Chuyển dữ liệu đầu vào thành DataFrame
    X = pd.DataFrame([input_data])

    # Lấy xác suất thuộc lớp 1 (tsunami)
    probability = float(model.predict_proba(X)[0, 1])

    # Áp dụng threshold đã lưu
    prediction = int(probability >= threshold)

    return {
        "probability": probability,
        "prediction": prediction,
        "threshold": threshold,
        "label": "Tsunami" if prediction == 1 else "No Tsunami"
    }