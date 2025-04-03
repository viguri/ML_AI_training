import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Create a dummy dataset
np.random.seed(0)
dummy_data = {
    'Feature1': np.random.normal(100, 10, 100).tolist() + [np.nan, 200],  # Normally distributed with an outlier
    'Feature2': np.random.randint(0, 100, 102).tolist(),  # Random integers
    'Category': ['A', 'B', 'C', 'D'] * 25 + [np.nan, 'A'],  # Categorical with some missing values
    'Target': np.random.choice([0, 1], 102).tolist()  # Binary target variable
}

# Convert the dictionary to a pandas DataFrame
df_dummy = pd.DataFrame(dummy_data)
# save the data a csv file
df_dummy.to_csv('dummy_data.csv', index=False)

# Display the first few rows of the dummy dataset
print("Original dataset:")
print(df_dummy.head())
print("\nMissing values:")
print(df_dummy.isnull().sum())

def load_data(df):
    return df

def handle_missing_values(df):
    # Handle numeric and categorical columns separately
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # For numeric columns, fill missing values with mean
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # For categorical columns, fill missing values with mode
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    return df

def remove_outliers(df):
    # Only remove outliers from numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        z_scores = np.abs(stats.zscore(df[numeric_cols]))
        df = df[(z_scores < 3).all(axis=1)]  # Remove rows with any outliers
    return df

def scale_data(df):
    # Only scale numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def encode_categorical(df, categorical_columns):
    # Only encode specified categorical columns that exist in the dataframe
    existing_cats = [col for col in categorical_columns if col in df.columns]
    if existing_cats:
        return pd.get_dummies(df, columns=existing_cats)
    return df

def save_data(df, output_filepath):
    df.to_csv(output_filepath, index=False)
    print(f"\nData saved to {output_filepath}")

# Load the data
df_preprocessed = load_data(df_dummy)

print("\nStarting preprocessing pipeline...")

# Handle missing values
print("\nHandling missing values...")
df_preprocessed = handle_missing_values(df_preprocessed)
print("Missing values after handling:")
print(df_preprocessed.isnull().sum())

# Remove outliers
print("\nRemoving outliers...")
df_preprocessed = remove_outliers(df_preprocessed)

# Scale the data
print("\nScaling numeric data...")
df_preprocessed = scale_data(df_preprocessed)

# Encode categorical variables
print("\nEncoding categorical variables...")
df_preprocessed = encode_categorical(df_preprocessed, ['Category'])

# Display the preprocessed data
print("\nPreprocessed dataset head:")
print(df_preprocessed.head())

# Save the cleaned and preprocessed DataFrame to a CSV file
save_data(df_preprocessed, 'preprocessed_dummy_data.csv')

print('\nPreprocessing complete!')
print(f"Original shape: {df_dummy.shape}")
print(f"Final shape: {df_preprocessed.shape}")

# Display the final dataset information
print("\nFinal dataset information:")
print(df_preprocessed.info())
print("\nMissing values in final dataset:")  
print(df_preprocessed.isnull().sum())
print("\nFinal dataset description:")   
print(df_preprocessed.describe())
print("\nFinal dataset head:")
print(df_preprocessed.head())
print("\nFinal dataset columns:")
print(df_preprocessed.columns)
