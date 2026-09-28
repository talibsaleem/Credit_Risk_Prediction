import numpy as np
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# 1. Processed Data Load Karna
data = np.load("data/processed_data.npz")
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

print(f"Original Training Class Distribution: 0={sum(y_train==0)}, 1={sum(y_train==1)}")

# 2. SMOTE Apply Karna (Only on Training Data to prevent Data Leakage)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Resampled Training Class Distribution: 0={sum(y_train_resampled==0)}, 1={sum(y_train_resampled==1)}")

# 3. Model 1: Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

print("\n=================== RANDOM FOREST RESULTS ===================")
print(classification_report(y_test, rf_preds))
print(f"Random Forest ROC-AUC Score: {roc_auc_score(y_test, rf_probs):.4f}")

# 4. Model 2: XGBoost Classifier (Industry Standard)
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)
xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

print("\n=================== XGBOOST RESULTS ===================")
print(classification_report(y_test, xgb_preds))
print(f"XGBoost ROC-AUC Score: {roc_auc_score(y_test, xgb_probs):.4f}")

# 5. Best Model Save Karna (XGBoost)
joblib.dump(xgb_model, "models/credit_risk_xgb_model.pkl")
print("\nBest Model (XGBoost) saved to 'models/credit_risk_xgb_model.pkl'")