import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load standardized data
df = pd.read_csv("cancer_data_standardized.csv")

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# Apply PCA (reduce to 2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Convert to DataFrame
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["target"] = y

# Save to CSV
df_pca.to_csv("cancer_data_pca.csv", index=False)

# Visualize PCA result
plt.figure(figsize=(8,6))
plt.scatter(df_pca["PC1"], df_pca["PC2"], c=df_pca["target"], cmap="coolwarm", alpha=0.6)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization of Breast Cancer Data")
plt.colorbar(label="Target")
plt.show()

print("PCA completed and saved as 'cancer_data_pca.csv'.")
