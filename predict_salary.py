"""Predict annual salary from employee details using the trained model."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model" / "salary_model.joblib"

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing. Run `python train_model.py` first.")

    model = joblib.load(MODEL_PATH)
    print("\nEMPLOYEE SALARY ESTIMATION")
    print("-------------------------")
    role = input("Job role (e.g. Software Engineer): ").strip()
    experience = float(input("Experience in years (0-40): "))
    hours = float(input("Working hours per week (1-100): "))

    if not role:
        raise ValueError("Job role cannot be empty.")
    if not 0 <= experience <= 40:
        raise ValueError("Experience must be between 0 and 40 years.")
    if not 1 <= hours <= 100:
        raise ValueError("Working hours must be between 1 and 100.")

    employee = pd.DataFrame([{
        "Job_Role": role,
        "Experience_Years": experience,
        "Working_Hours_Per_Week": hours,
    }])
    estimate = float(model.predict(employee)[0])
    print(f"\nEstimated annual salary: ₹{estimate:,.0f}")
    print("Note: This is an educational estimate, not a salary offer or guarantee.")

if __name__ == "__main__":
    main()
