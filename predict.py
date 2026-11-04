from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent
TEST_FILE = BASE_DIR / "dataset" / "GUIDE_Test.csv"
MODEL_FILE = BASE_DIR / "models" / "random_forest.joblib"
COLUMNS_FILE = BASE_DIR / "models" / "feature_columns.joblib"

ID_AS_CATEGORY = ["OrgId", "DetectorId", "AlertTitle",
                  "ApplicationId", "OAuthApplicationId",
                  "ApplicationName", "OSFamily", "OSVersion",
                  "CountryCode", "State", "City"]


def to_text(v):
    """123 aur 123.0 ko same text 123 banao. Missing 'nan' ban jata hai."""
    if pd.isna(v):
        return "nan"
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(v)
    except (TypeError, ValueError):
        return str(v)


print("=" * 55)
print("CYBERSECURITY INCIDENT PREDICTION DEMO")
print("=" * 55)

if not MODEL_FILE.exists() or not COLUMNS_FILE.exists():
    raise FileNotFoundError("Trained model not found. Run python main.py first.")
if not TEST_FILE.exists():
    raise FileNotFoundError(f"Test dataset not found: {TEST_FILE}")

model = joblib.load(MODEL_FILE)
feature_columns = joblib.load(COLUMNS_FILE)

row = pd.read_csv(TEST_FILE, nrows=1, low_memory=False)

# main.py jaisa hi time features
if "Timestamp" in row.columns:
    t = pd.to_datetime(row["Timestamp"], errors="coerce", utc=True)
    row["Hour"] = t.dt.hour
    row["DayOfWeek"] = t.dt.day_of_week
    row["Month"] = t.dt.month

# main.py jaisa hi ID -> text step
for c in ID_AS_CATEGORY:
    if c in row.columns:
        row[c] = row[c].map(to_text)

# sirf wahi columns, wahi order me, jo training me the
X_new = row.reindex(columns=feature_columns)

prediction = model.predict(X_new)[0]
print("\nPredicted grade:", prediction)
if "IncidentGrade" in row.columns:
    print("Actual grade   :", row["IncidentGrade"].iloc[0])

probabilities = model.predict_proba(X_new)[0]
print("\nClass probabilities:")
for label, p in zip(model.classes_, probabilities):
    print(f"{label}: {p:.2%}")

print("\nThis is a demo prediction, not a performance evaluation.")