import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# 1. Load dataset (using 50,000 rows for fast training)
print("1. Loading dataset...")
df = pd.read_csv("dataset/GUIDE_Train.csv", nrows=50000, low_memory=False)

# 2. Handle missing target values and duplicates
print("2. Handling missing values...")
df = df.dropna(subset=["IncidentGrade"]).drop_duplicates()

# 3. Separate input features (X) and target (y)
print("3. Selecting features and target...")
y = df["IncidentGrade"]
groups = df["IncidentId"]

# Exclude identifier columns to prevent memorization
drop_cols = [
    "Id", "IncidentId", "AlertId", "IncidentGrade",
    "DeviceId", "DeviceName", "AccountSid", "AccountUpn",
    "AccountObjectId", "AccountName", "IpAddress", "Sha256",
    "Url", "NetworkMessageId", "EmailClusterId", "FileName",
    "FolderPath", "RegistryKey", "RegistryValueName",
    "RegistryValueData", "ResourceIdName", "ActionGrouped",
    "ActionGranular", "Usage"
]
X = df.drop(columns=drop_cols, errors="ignore")

# Extract simple temporal features from Timestamp
if "Timestamp" in X.columns:
    t = pd.to_datetime(X["Timestamp"], errors="coerce")
    X["Hour"], X["DayOfWeek"], X["Month"] = t.dt.hour, t.dt.day_of_week, t.dt.month
    X = X.drop(columns=["Timestamp"], errors="ignore")

# Treat code-like columns as categories, not numbers
id_cols = [
    "OrgId", "DetectorId", "AlertTitle", "ApplicationId",
    "OAuthApplicationId", "ApplicationName", "OSFamily",
    "OSVersion", "CountryCode", "State", "City"
]
for col in id_cols:
    if col in X.columns:
        X[col] = X[col].astype(str)

# Identify numeric and categorical columns
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
joblib.dump(X.columns.tolist(), "models/feature_columns.joblib")

# 4. Split data into training (80%) and testing (20%)
# We use GroupShuffleSplit on IncidentId to keep all alerts of an incident together
# in either train or test, which prevents data leakage.
print("4. Splitting dataset...")
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# 5. Build preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])

# 6. Train Logistic Regression
print("5. Training Logistic Regression...")
lr = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
lr.fit(X_train, y_train)

# 7. Train Random Forest
print("6. Training Random Forest...")
rf = Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(n_estimators=100, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1))])
rf.fit(X_train, y_train)

# 8. Evaluate both models
print("7. Evaluating models...")
results = []
reports = {}

for name, model in [("Random Forest", rf), ("Logistic Regression", lr)]:
    preds = model.predict(X_test)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)

    results.append({
        "Model": name,
        "Train Accuracy": train_acc,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Weighted F1-score": weighted_f1,
        "Macro F1-score": macro_f1
    })
    reports[name] = classification_report(y_test, preds, zero_division=0)
    print(f"{name} -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {weighted_f1:.4f}")

# 9. Save comparison table and reports
comparison_df = pd.DataFrame(results)
comparison_df.to_csv("outputs/model_comparison.csv", index=False)

with open("outputs/model_reports.txt", "w", encoding="utf-8") as f:
    for name, rpt in reports.items():
        f.write(f"\n{name}\n" + "=" * 45 + f"\n{rpt}\n")

# 10. Save visualizations
labels = sorted(y.unique())

# Model comparison bar chart
plt.figure(figsize=(8, 5))
comparison_df.set_index("Model")[["Accuracy", "Precision", "Recall", "Weighted F1-score"]].plot(kind="bar", ylim=(0, 1))
plt.title("Model Comparison")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=300)
plt.close()

# Class distribution chart
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="IncidentGrade", order=labels)
plt.title("Incident Grade Distribution")
plt.xlabel("Incident Grade")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/class_distribution.png", dpi=300)
plt.close()

# Confusion matrices
for name, model, fname in [("Random Forest", rf, "confusion_matrix.png"), ("Logistic Regression", lr, "logistic_regression_confusion_matrix.png")]:
    cm = confusion_matrix(y_test, model.predict(X_test), labels=labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"outputs/{fname}", dpi=300)
    plt.close()

# 11. Save trained models using Joblib
print("8. Saving trained models...")
joblib.dump(lr, "models/logistic_regression.joblib")
joblib.dump(rf, "models/random_forest.joblib")
print("Training completed successfully! Models and outputs are saved.")
