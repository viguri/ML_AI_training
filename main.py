import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
import sys

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"\nDataset loaded successfully from {filepath}")
        print(f"Shape: {df.shape}")
        print("\nColumns:", df.columns.tolist())
        print("\nData types:")
        print(df.dtypes)
        print("\nMissing values:")
        print(df.isnull().sum())
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

def handle_missing_values(df):
    print("\nHandling missing values...")
    
    # Handle numeric and categorical columns separately
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # For numeric columns, fill missing values with mean
    if len(numeric_cols) > 0:
        print("Filling numeric columns with mean values...")
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # For categorical columns, fill missing values with mode
    if len(categorical_cols) > 0:
        print("Filling categorical columns with mode values...")
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    print("\nMissing values after handling:")
    print(df.isnull().sum())
    return df

def remove_outliers(df):
    print("\nRemoving outliers...")
    # Only remove outliers from numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        original_shape = df.shape
        z_scores = np.abs(stats.zscore(df[numeric_cols]))
        df = df[(z_scores < 3).all(axis=1)]
        print(f"Removed {original_shape[0] - df.shape[0]} rows containing outliers")
    return df

def scale_data(df):
    print("\nScaling numerical data...")
    # Only scale numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        print(f"Scaled {len(numeric_cols)} numeric columns")
    return df

def encode_categorical(df, categorical_columns):
    print("\nEncoding categorical variables...")
    # Only encode specified categorical columns that exist in the dataframe
    existing_cats = [col for col in categorical_columns if col in df.columns]
    if existing_cats:
        original_cols = df.columns.tolist()
        df = pd.get_dummies(df, columns=existing_cats)
        new_cols = [col for col in df.columns if col not in original_cols]
        print(f"Encoded {len(existing_cats)} categorical columns, creating {len(new_cols)} new features")
    return df

def save_data(df, output_filepath):
    try:
        df.to_csv(output_filepath, index=False)
        print(f"\nData successfully saved to {output_filepath}")
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        sys.exit(1)

def main():
    input_file = 'your_dataset.csv'
    output_file = 'cleaned_preprocessed_data.csv'
    
    print("Starting data preprocessing pipeline...")
    
    # Load and process data
    df = load_data(input_file)
    original_shape = df.shape
    
    # Process the data
    df = handle_missing_values(df)
    df = remove_outliers(df)
    df = scale_data(df)
    
    # Identify categorical columns (columns with object dtype)
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        df = encode_categorical(df, categorical_columns)
    
    # Save the processed data
    save_data(df, output_file)
    
    print("\nData preprocessing completed successfully!")
    print(f"Original shape: {original_shape}")
    print(f"Final shape: {df.shape}")
    print(f"Final number of features: {len(df.columns)}")

if __name__ == "__main__":
    main()