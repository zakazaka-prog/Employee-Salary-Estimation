"""Train and evaluate the employee salary regression model."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "dataset" / "employee_salary.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "salary_model.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"

FEATURES = ["Job_Role", "Experience_Years", "Working_Hours_Per_Week"]
TARGET = "Annual_Salary_INR"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run `python generate_dataset.py` first."
        )

    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    df = df.dropna(subset=required)
    X = df[FEATURES]
    y = df[TARGET]

    preprocessor = ColumnTransformer(
        transformers=[
            ("job_role", OneHotEncoder(handle_unknown="ignore"), ["Job_Role"])
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "model": "RandomForestRegressor",
        "test_rows": int(len(X_test)),
        "mae_inr": round(float(mean_absolute_error(y_test, predictions)), 2),
        "rmse_inr": round(float(mean_squared_error(y_test, predictions) ** 0.5), 2),
        "r2_score": round(float(r2_score(y_test, predictions)), 4),
        "dataset_note": "Synthetic demonstration data; not real market salary data.",
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    print("Model trained and saved.")
    print(json.dumps(metrics, indent=4))
    print(f"Model: {MODEL_PATH}")

if __name__ == "__main__":
    main()
