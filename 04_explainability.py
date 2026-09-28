import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Model aur Feature Names Load Karna
model = joblib.load("models/credit_risk_xgb_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# 2. Test Data Load Karna
data = np.load("data/processed_data.npz", allow_pickle=True)
X_test = data['X_test']

# DataFrame banana feature names ke saath taake plots readable hon
X_test_df = pd.DataFrame(X_test, columns=feature_names)

# 3. SHAP Explainer Initialize Karna
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test_df)

print("Calculating SHAP Feature Importance...")

# 4. Global Feature Importance Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_df, show=False)
plt.title("SHAP Feature Importance (Credit Risk Drivers)", fontsize=14)
plt.tight_layout()
plt.savefig("models/shap_summary.png")
plt.show()

# 5. Local Explanation (Single Customer Example)
customer_idx = 0  # Pehle customer ki prediction ki explanation
plt.figure(figsize=(8, 4))
shap.plots.waterfall(shap_values[customer_idx], show=False)
plt.title(f"Credit Risk Explanation for Customer #{customer_idx}", fontsize=12)
plt.tight_layout()
plt.savefig("models/shap_customer_waterfall.png")
plt.show()

print("\nExplainability Plots successfully generated and saved in 'models/' folder!")