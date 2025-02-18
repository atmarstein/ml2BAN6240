import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load the dataset
cancer = load_breast_cancer()

# Convert to DataFrame
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target

# Save to CSV
df.to_csv("cancer_data.csv", index=False)

print("Dataset successfully loaded and saved as 'cancer_data.csv'.")
