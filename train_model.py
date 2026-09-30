import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the real data sample
df = pd.read_csv("us_sample.csv")

# 2. Convert the text column (weather) into numbers
df_encoded = pd.get_dummies(df, columns=["weather"])

# 3. Inputs (X) and answer (y)
X = df_encoded.drop("risk", axis=1)
y = df_encoded["risk"]

# 4. Split: 80% learning, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train (smaller trees so the file is small)
model = RandomForestClassifier(
    n_estimators=100, max_depth=20, min_samples_leaf=5,
    n_jobs=-1, random_state=42
)
model.fit(X_train, y_train)

# 6. Test
predictions = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print()
print(classification_report(y_test, predictions))

# 7. Show which inputs matter most
importance = pd.Series(model.feature_importances_, index=X.columns)
print("Top 8 important features:")
print(importance.sort_values(ascending=False).head(8))

# 8. Save (compressed)
joblib.dump(model, "model.pkl", compress=3)
joblib.dump(list(X.columns), "columns.pkl")
print()
print("Model saved as model.pkl")