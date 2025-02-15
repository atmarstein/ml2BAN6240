# Principal Component Analysis (PCA) on Breast Cancer Dataset  

## **Project Overview**  
This project applies **Principal Component Analysis (PCA)** to the **Breast Cancer dataset** from `sklearn.datasets` to identify essential variables. The dataset is reduced to **2 principal components**, and a **logistic regression model** (optional) is implemented to classify cancer diagnoses.

## **Project Files**  
- `load_data.py` → Loads the dataset and saves it as a CSV file.  
- `preprocess_data.py` → Standardizes the dataset using `StandardScaler`.  
- `pca_analysis.py` → Performs PCA, reduces to **2 components**, and visualizes results.  
- `logistic_regression.py` (Optional) → Implements logistic regression for classification.  

## **Setup and Installation**  
### **1. Install Required Libraries**  
Before running the scripts, install the necessary dependencies using:  
```bash
pip install pandas scikit-learn matplotlib
```

### **2. Run the Scripts in Order**  
Run each script sequentially in **Thonny** or a terminal:  
```bash
python load_data.py
python preprocess_data.py
python pca_analysis.py
python logistic_regression.py  # Optional
```

## **How the Project Works**  
1. **Load Dataset**  
   - Uses `sklearn.datasets` to load the **Breast Cancer dataset**.  
   - Saves the data as `cancer_data.csv`.  

2. **Preprocess Data**  
   - Standardizes features using `StandardScaler`.  
   - Saves the processed data as `cancer_data_standardized.csv`.  

3. **Perform PCA**  
   - Reduces dimensions to **2 principal components**.  
   - Saves the PCA-transformed data as `cancer_data_pca.csv`.  
   - Plots a **scatter plot** to visualize PCA results.  

4. **Bonus: Logistic Regression (Optional)**  
   - Trains a **logistic regression model** on the PCA-reduced data.  
   - Splits data into **training (80%)** and **testing (20%)**.  
   - Evaluates and prints **classification accuracy**.  

## **Expected Outputs**  
1. **PCA Scatter Plot:** A visual representation of **two principal components**, with color-coded cancer diagnoses.  
2. **Logistic Regression Accuracy:** Displays model accuracy (e.g., `Accuracy: 0.92`).  

