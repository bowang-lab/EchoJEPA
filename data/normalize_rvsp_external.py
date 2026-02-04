import pandas as pd
import pickle
import numpy as np

# 1. Load your new external dataset
# assuming it has a column 'Value' with the raw RVSP in mmHg (e.g., 35.0)
df_external = pd.read_csv('external_hospital_rvsp_data.csv')

# 2. Clean the data first (match the logic from your rvsp training script)
# Unlike LVEF (0-100%), RVSP is a pressure in mmHg. 
# Based on your training stats (Min ~10, Max ~164), I've set safe bounds of 0 to 200.
df_external = df_external[(df_external['Value'] >= 0) & (df_external['Value'] <= 200)].copy()

# 3. Load the scaler learned from the ORIGINAL training data
# This contains the specific Mean (~34.465) and Std (~14.013) from your training set
with open('rvsp_scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

# 4. Transform the new values
# Note: reshape is required because scaler expects a 2D array
external_values = df_external['Value'].values.reshape(-1, 1)

# Apply the transformation
df_external['norm_value'] = loaded_scaler.transform(external_values)

# --- Verification ---
# Now 'norm_value' is your Ground Truth for calculating test error
print("Data Preview:")
print(df_external[['Value', 'norm_value']].head())

print("\nQuick Check:")
print(f"Mean of input data: {df_external['Value'].mean():.2f} mmHg")
print(f"Mean of normalized data: {df_external['norm_value'].mean():.4f} (Should be close to 0 if external data is similar to training data)")