import pandas as pd
import pickle
import numpy as np

# 1. Load your new external dataset
# assuming it has a column 'Value' with the raw LVEF percentage (e.g., 55.0)
df_external = pd.read_csv('external_hospital_data.csv')

# 2. Clean the data first (match the logic from lvef.py)
# The training script filtered values to be strictly 0-100
df_external = df_external[(df_external['Value'] >= 0) & (df_external['Value'] <= 100)].copy()

# 3. Load the scaler learned from the ORIGINAL training data
with open('lvef_scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

# 4. Transform the new values
# Note: reshape is required because scaler expects a 2D array
external_values = df_external['Value'].values.reshape(-1, 1)

# Apply the transformation
df_external['norm_value'] = loaded_scaler.transform(external_values)

# Now 'norm_value' is your Ground Truth for calculating test error
print(df_external[['Value', 'norm_value']].head())