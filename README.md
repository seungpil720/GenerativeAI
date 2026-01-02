# GenerativeAI

Sure, here's a consolidated version of the code so far, formatted for clarity and ready for upload to a GitHub repository.

### 1. Project Setup
Create a new directory for your project and initialize a Git repository:
```bash
mkdir NHANES_GenerativeAI
cd NHANES_GenerativeAI
git init
```

### 2. Create a `requirements.txt` file
Include the necessary Python packages:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
openpyxl
```

### 3. Create a `data` directory
Place your NHANES data file (`KNHANES5_VitaD.xlsx`) in the `data` directory.

### 4. Create the Main Script: `main.py`
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'data/KNHANES5_VitaD.xlsx'
data_sample = pd.read_excel(file_path, nrows=100)

# Define columns of interest
columns_of_interest = [
    'DJ4_dg', 'DJ4_ag', 'DJ4_pr', 'DJ4_pt',
    'DL1_dg', 'DL1_ag', 'DL1_pr', 'DL1_pt',
    'sex', 'age', 'edu', 'incm', 'ho_incm',
    'region', 'year', 'HE_BMI', 'HE_obe',
    'sm_presnt', 'pa_high', 'pa_walk',
    'N_EN', 'N_PROT', 'N_FAT', 'N_CHO', 'N_FIBER',
    'N_CA', 'N_FE', 'N_VA', 'N_VITC'
]

# Filter and clean the data
filtered_data = data_sample[columns_of_interest]
filtered_data_cleaned = filtered_data.dropna()

# Encode categorical variables
label_encoder = LabelEncoder()
for col in ['sex', 'edu', 'sm_presnt', 'pa_high', 'pa_walk', 'region', 'HE_obe']:
    filtered_data_cleaned[col] = label_encoder.fit_transform(filtered_data_cleaned[col])

# Normalize continuous variables
continuous_vars = [
    'age', 'incm', 'ho_incm', 'HE_BMI',
    'N_EN', 'N_PROT', 'N_FAT', 'N_CHO', 'N_FIBER',
    'N_CA', 'N_FE', 'N_VA', 'N_VITC'
]
scaler = StandardScaler()
filtered_data_cleaned[continuous_vars] = scaler.fit_transform(filtered_data_cleaned[continuous_vars])

# Split the data into training, validation, and test sets
train_data, test_data = train_test_split(filtered_data_cleaned, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# PCA for dimensionality reduction
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data)
pca = PCA(n_components=10)
train_data_pca = pca.fit_transform(train_data_scaled)
synthetic_data_pca = pca.inverse_transform(train_data_pca[:test_data.shape[0]])

# Create a synthetic data dataframe
synthetic_data_df = pd.DataFrame(scaler.inverse_transform(synthetic_data_pca), columns=filtered_data_cleaned.columns)

# Statistical comparison
real_means = train_data.mean()
synthetic_means = synthetic_data_df.mean()
real_vars = train_data.var()
synthetic_vars = synthetic_data_df.var()
comparison_df = pd.DataFrame({
    'Real Mean': real_means,
    'Synthetic Mean': synthetic_means,
    'Real Variance': real_vars,
    'Synthetic Variance': synthetic_vars
})

# Display the comparison dataframe
print("Statistical Comparison:\n", comparison_df.head())

# Correlation matrices
real_corr_matrix = train_data.corr()
synthetic_corr_matrix = pd.DataFrame(synthetic_data_pca, columns=filtered_data_cleaned.columns).corr()

# Plot correlation matrices
plt.figure(figsize=(12, 6))

# Real data correlation matrix
plt.subplot(1, 2, 1)
sns.heatmap(real_corr_matrix, annot=False, cmap='coolwarm')
plt.title('Real Data Correlation Matrix')

# Synthetic data correlation matrix
plt.subplot(1, 2, 2)
sns.heatmap(synthetic_corr_matrix, annot=False, cmap='coolwarm')
plt.title('Synthetic Data Correlation Matrix')

plt.tight_layout()
plt.show()

# Trend Analysis: Prevalence of asthma and allergies with age
plt.figure(figsize=(10, 5))

# Asthma prevalence by age
plt.subplot(1, 2, 1)
sns.histplot(train_data[train_data['DJ4_dg'] == 1]['age'], bins=10, kde=True)
plt.title('Asthma Prevalence by Age')
plt.xlabel('Age')
plt.ylabel('Count')

# Allergy prevalence by age
plt.subplot(1, 2, 2)
sns.histplot(train_data[train_data['DL1_dg'] == 1]['age'], bins=10, kde=True)
plt.title('Allergy Prevalence by Age')
plt.xlabel('Age')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# Asthma prevalence by gender
asthma_by_gender = train_data.groupby('sex')['DJ4_dg'].mean()

# Plot asthma prevalence by gender
plt.figure(figsize=(8, 5))
sns.barplot(x=asthma_by_gender.index, y=asthma_by_gender.values, palette='viridis')
plt.title('Asthma Prevalence by Gender')
plt.xlabel('Gender (0: Male, 1: Female)')
plt.ylabel('Prevalence (Proportion of Participants Diagnosed)')
plt.show()
```
