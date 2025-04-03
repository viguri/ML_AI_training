import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import missingno as msno  # Optional: for visualizing missing data

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import missingno as msno  # Optional: for visualizing missing data

# Load your dataset into a pandas DataFrame
df = pd.read_csv('your_dataset.csv')  # Replace 'your_dataset.csv' with your actual file path

# Display the first few rows of the dataset
print(df.head())

# Visualize missing data (optional)
msno.matrix(df)
msno.heatmap(df)

# Drop rows with missing values
df_cleaned = df.dropna()

# Or, fill missing values with the mean
df_filled = df.fillna(df.mean())

# Identify outliers using Z-score
from scipy import stats

z_scores = np.abs(stats.zscore(df_cleaned))
df_no_outliers = df_cleaned[(z_scores < 3).all(axis=1)]

# Or cap outliers at a threshold
upper_limit = df_cleaned['column_name'].quantile(0.95)
df_cleaned['column_name'] = np.where(df_cleaned['column_name'] > upper_limit, upper_limit, df_cleaned['column_name'])

# Min-Max Scaling
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned), columns=df_cleaned.columns)

# Z-score Standardization
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df_cleaned), columns=df_cleaned.columns)

# One-hot encoding for categorical variables
df_encoded = pd.get_dummies(df_scaled, columns=['categorical_column_name'])

# Save the cleaned and preprocessed DataFrame to a new CSV file  
df_encoded.to_csv('cleaned_preprocessed_data.csv', index=False)

print('Data cleaning and preprocessing complete. File saved as cleaned_preprocessed_data.csv')


