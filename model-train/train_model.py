import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# 1. Load data
data = pd.read_csv("data.csv")
print("Original dataset shape:", data.shape)

# 🔍 1.1 Remove rows with missing values
data = data.dropna().reset_index(drop=True)
print("Cleaned dataset shape (no NaNs):", data.shape)

# 2. Separate features and labels
X = data.drop("label", axis=1)
y = data["label"]

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluate
y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. Save model
with open("sign_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("\n✅ Model trained and saved to 'sign_model.pkl'")