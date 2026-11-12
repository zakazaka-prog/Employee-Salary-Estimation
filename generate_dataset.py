"""Generate a clearly labeled synthetic employee salary dataset for demonstration."""
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42
ROWS = 1500

ROLE_BASE = {
    "Software Engineer": 420_000,
    "Data Analyst": 360_000,
    "Web Developer": 330_000,
    "Data Scientist": 520_000,
    "Machine Learning Engineer": 560_000,
    "Manager": 600_000,
    "UI/UX Designer": 340_000,
    "Network Engineer": 370_000,
}

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    roles = list(ROLE_BASE)
    records = []

    for _ in range(ROWS):
        role = rng.choice(roles)
        experience = int(rng.integers(0, 21))
        hours = int(rng.integers(30, 61))

        # Synthetic rule used only to create practice data, not a real salary survey.
        experience_component = experience * rng.uniform(38_000, 52_000)
        hours_component = (hours - 40) * 2_500
        noise = rng.normal(0, 45_000)
        salary = max(180_000, ROLE_BASE[role] + experience_component + hours_component + noise)

        records.append({
            "Job_Role": role,
            "Experience_Years": experience,
            "Working_Hours_Per_Week": hours,
            "Annual_Salary_INR": round(salary, 2),
        })

    df = pd.DataFrame(records)
    output = Path(__file__).parent / "dataset" / "employee_salary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Created {len(df)} synthetic records: {output}")

if __name__ == "__main__":
    main()
