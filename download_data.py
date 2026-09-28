import pandas as pd
import os

# 1. Directory check / creation
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print("Dataset download ho raha hai, bara-e-karam thoda wait karein...")

# 2. UCI ML Repository se direct clean dataset fetch karna
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# Column names standard definitions ke mutabiq
columns = [
    'status_existing_checking', 'duration_in_month', 'credit_history', 'purpose',
    'credit_amount', 'savings_account', 'present_employment_since', 
    'installment_rate_in_percentage_of_disposable_income', 'personal_status_sex',
    'other_debtors_or_guarantors', 'present_residence_since', 'property',
    'age', 'other_installment_plans', 'housing', 'number_of_existing_credits',
    'job', 'number_of_people_being_liable_to_provide_maintenance_for',
    'telephone', 'foreign_worker', 'credit_risk'
]

# Read CSV (Space delimited)
df = pd.read_csv(url, sep=' ', names=columns)

# Target variable transformation (UCI dataset mein 1 = Good, 2 = Bad Default hota hai)
# Standard Machine Learning ke liye: 0 = Good Credit, 1 = Default Risk
df['credit_risk'] = df['credit_risk'].map({1: 0, 2: 1})

# Save to data folder
file_path = os.path.join(data_dir, "credit_data.csv")
df.to_csv(file_path, index=False)

print(f"Success! Dataset aap ke '{file_path}' location par save ho gaya hai.")
print(f"Dataset Dimensions: {df.shape[0]} Rows, {df.shape[1]} Columns")