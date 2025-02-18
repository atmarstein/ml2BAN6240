import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("cancer_data.csv")

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
df_scaled["target"] = y  # Add target back

# Save to CSV
df_scaled.to_csv("cancer_data_standardized.csv", index=False)

print("Data successfully standardized and saved as 'cancer_data_standardized.csv'.")
