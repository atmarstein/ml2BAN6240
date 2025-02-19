import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load and Save the Dataset
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df["target"] = cancer.target
df.to_csv("cancer_data.csv", index=False)
print("Dataset successfully loaded and saved as 'cancer_data.csv'.")

# Step 2: Standardize the Data
df = pd.read_csv("cancer_data.csv")
X = df.drop(columns=["target"])
y = df["target"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
df_scaled["target"] = y
df_scaled.to_csv("cancer_data_standardized.csv", index=False)
print("Data successfully standardized and saved as 'cancer_data_standardized.csv'.")

# Step 3: Apply PCA
df = pd.read_csv("cancer_data_standardized.csv")
X = df.drop(columns=["target"])
y = df["target"]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["target"] = y
df_pca.to_csv("cancer_data_pca.csv", index=False)
print("PCA completed and saved as 'cancer_data_pca.csv'.")

# Step 4: Visualize PCA Result
plt.figure(figsize=(8, 6))
plt.scatter(df_pca["PC1"], df_pca["PC2"], c=df_pca["target"], cmap="coolwarm", alpha=0.6)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization of Breast Cancer Data")
plt.colorbar(label="Target")
plt.show()

# Step 5: Train Logistic Regression Model
df = pd.read_csv("cancer_data_pca.csv")
X = df[["PC1", "PC2"]]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.2f}")
