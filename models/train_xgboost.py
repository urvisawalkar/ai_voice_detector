import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


# ============================================
# Load dataset
# ============================================
df = pd.read_csv("../data/features.csv")

print("Dataset size:", df.shape)


# ============================================
# Split features and labels
# ============================================
X = df.drop("label", axis=1)
y = df["label"]


# ============================================
# Train/test split
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================
# Train XGBoost model
# ============================================
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05
)

model.fit(X_train, y_train)


# ============================================
# Evaluate
# ============================================
preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)

print("\n✅ Accuracy:", acc)
print("\nClassification Report:\n")
print(classification_report(y_test, preds))


# ============================================
# Save model
# ============================================
joblib.dump(model, "forensic_model.pkl")

print("\nModel saved as models/forensic_model.pkl")
