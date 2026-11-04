from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# ---------- 1. Paths aur settings ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "dataset" / "GUIDE_Train.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MAX_ROWS = 50000
target = "IncidentGrade"

# Ye columns model ko nahi dene (identifiers / memorization risk / post-triage actions)
EXCLUDE = [
    "Id", "IncidentId", "AlertId", target,
    "ActionGrouped", "ActionGranular",
    "DeviceId", "DeviceName",
    "AccountSid", "AccountUpn", "AccountObjectId", "AccountName",
    "IpAddress", "Sha256", "Url", "NetworkMessageId", "EmailClusterId",
    "FileName", "FolderPath",
    "RegistryKey", "RegistryValueName", "RegistryValueData",
    "ResourceIdName", "Usage"
]

# Ye numbers jaise dikhte hain par asal me categories/codes hain, isliye text bana denge
ID_AS_CATEGORY = ["OrgId", "DetectorId", "AlertTitle",
                  "ApplicationId", "OAuthApplicationId",
                  "ApplicationName", "OSFamily", "OSVersion",
                  "CountryCode", "State", "City"]

print("=" * 55)
print("CYBERSECURITY INCIDENT PREDICTION SYSTEM")
print("=" * 55)


def add_time_features(data):
    """Timestamp se Hour, DayOfWeek, Month banao, phir Timestamp hata do."""
    data = data.copy()
    if "Timestamp" in data.columns:
        t = pd.to_datetime(data["Timestamp"], errors="coerce", utc=True)
        data["Hour"] = t.dt.hour
        data["DayOfWeek"] = t.dt.day_of_week
        data["Month"] = t.dt.month
        data = data.drop(columns=["Timestamp"])
    return data


def to_text(v):
    """123 aur 123.0 ko same text 123 banao. Missing 'nan' ban jata hai."""
    if pd.isna(v):
        return "nan"
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(v)
    except (TypeError, ValueError):
        return str(v)


def id_columns_to_text(data):
    """ID/code jaise numeric columns ko category (text) bana do."""
    data = data.copy()
    for c in ID_AS_CATEGORY:
        if c in data.columns:
            data[c] = data[c].map(to_text)
    return data


# ---------- 2. Data load ----------
df = pd.read_csv(DATA_FILE, low_memory=False)
df = df.dropna(subset=[target])
before = len(df)
df = df.drop_duplicates()
print("Duplicate rows removed:", before - len(df))

if len(df) > MAX_ROWS:
    df = df.sample(MAX_ROWS, random_state=42)

print("Rows used:", len(df))

# ---------- 3. Features (X), target (y), groups ----------
groups = df["IncidentId"]

removed = [c for c in EXCLUDE if c in df.columns]
print("Excluded columns found in data:", removed)

X = df.drop(columns=EXCLUDE, errors="ignore")
y = df[target]

X = add_time_features(X)
X = id_columns_to_text(X)

empty_columns = X.columns[X.isna().all()].tolist()
X = X.drop(columns=empty_columns)

numeric_columns = X.select_dtypes(include="number").columns.tolist()
categorical_columns = X.select_dtypes(exclude="number").columns.tolist()

print("Numeric features:", len(numeric_columns), numeric_columns)
print("Categorical features:", len(categorical_columns), categorical_columns)
print("Classes:", sorted(y.unique()))

# ---------- 4. Split (IncidentId ke basis par) ----------
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
print("Train rows:", len(X_train), "| Test rows:", len(X_test))
print("Incidents common in train and test:", len(overlap))


# ---------- 5. Preprocessing ----------
def make_preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_columns),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                      min_frequency=20))
        ]), categorical_columns)
    ])


# ---------- 6. Models ----------
models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=100, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1))
    ])
}

# ---------- 7. Train aur evaluate ----------
labels = sorted(y.unique())
results = []
reports = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))

    results.append({
        "Model": name,
        "Train Accuracy": train_accuracy,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "Weighted F1-score": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "Macro F1-score": f1_score(y_test, predictions, average="macro", zero_division=0)
    })
    reports[name] = classification_report(y_test, predictions, zero_division=0)

    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, MODEL_DIR / filename)

    cm = confusion_matrix(y_test, predictions, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename.replace(".joblib", "_confusion_matrix.png"), dpi=300)
    plt.close()

joblib.dump(X.columns.tolist(), MODEL_DIR / "feature_columns.joblib")

# ---------- 8. Comparison table aur graph (Macro-F1 se sort) ----------
comparison = pd.DataFrame(results).sort_values("Macro F1-score", ascending=False)
comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

comparison.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "Weighted F1-score", "Macro F1-score"]
].plot(kind="bar", figsize=(10, 6))
plt.title("Logistic Regression vs Random Forest")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=300)
plt.close()

# ---------- 9. Class distribution graph ----------
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x=target, order=labels)
plt.title("Incident Grade Distribution")
plt.xlabel("Incident Grade")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=300)
plt.close()

# ---------- 10. Random Forest feature importance (asli column ke naam se) ----------
rf_model = models["Random Forest"]
encoded_names = rf_model.named_steps["preprocessor"].get_feature_names_out()
importances = rf_model.named_steps["classifier"].feature_importances_


def original_name(encoded):
    name = encoded.split("__", 1)[1]
    for col in sorted(X.columns, key=len, reverse=True):
        if name == col or name.startswith(col + "_"):
            return col
    return name


imp = pd.Series(importances, index=[original_name(n) for n in encoded_names])
top = imp.groupby(level=0).sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(9, 6))
top[::-1].plot(kind="barh")
plt.title("Random Forest - Top 10 Important Features")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rf_feature_importance.png", dpi=300)
plt.close()

# ---------- 11. Reports save ----------
with open(OUTPUT_DIR / "model_reports.txt", "w", encoding="utf-8") as file:
    for name, report in reports.items():
        file.write(f"\n{name}\n")
        file.write("=" * 45 + "\n")
        file.write(report + "\n")

print("\nSaved outputs in:", OUTPUT_DIR)
print("Saved models in:", MODEL_DIR)
print("Model comparison completed successfully.")