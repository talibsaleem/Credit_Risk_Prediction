import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

# 1. Dataset Load
df = pd.read_csv("data/credit_data.csv")

# 2. Features (X) aur Target (y) ko alag karna
X = df.drop(columns=['credit_risk'])
y = df['credit_risk']

# 3. Categorical Columns ko One-Hot Encode karna
# drop_first=True taake Dummy Variable Trap / Multicollinearity se bacha ja sake
X_encoded = pd.get_dummies(X, drop_first=True)

print(f"Original Feature Count: {X.shape[1]}")
print(f"Encoded Feature Count: {X_encoded.shape[1]}")

# 4. Train-Test Split (80% Training, 20% Testing)
# stratify=y rakha hai taake Train aur Test dono mein 70/30 class ratio barkarar rahe
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# 5. Clean NumPy Array Conversion & Save (UPDATED HERE)
np.savez(
    "data/processed_data.npz", 
    X_train=np.array(X_train, dtype=np.float32), 
    X_test=np.array(X_test, dtype=np.float32), 
    y_train=np.array(y_train, dtype=np.int32), 
    y_test=np.array(y_test, dtype=np.int32)
)

joblib.dump(list(X_encoded.columns), "models/feature_names.pkl")

print("Data Preprocessing Successful!")