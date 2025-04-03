# Data Preprocessing Pipeline

This project provides a robust data preprocessing pipeline implemented in Python. It includes two main scripts that demonstrate the functionality with both real and synthetic data.

## Features

- **Missing Value Handling**
  - Numeric data: Imputed with mean values
  - Categorical data: Imputed with mode values (most frequent)
  - Fallback to 'Unknown' for categorical columns with no mode

- **Outlier Removal**
  - Uses Z-score method to detect outliers
  - Removes data points that are more than 3 standard deviations from the mean
  - Only applies to numeric columns

- **Data Scaling**
  - Standardizes numeric features using StandardScaler
  - Transforms features to zero mean and unit variance
  - Preserves categorical columns

- **Categorical Encoding**
  - Converts categorical variables to dummy/indicator variables
  - Uses one-hot encoding for categorical columns
  - Automatically detects categorical columns

## Files

### main.py
The primary script for processing real data files. It:
- Loads data from a CSV file
- Applies the complete preprocessing pipeline
- Provides detailed progress feedback
- Saves the processed data to a new CSV file

Usage:
```bash
python main.py
```

### dummy.py
A demonstration script that:
- Creates synthetic data with known properties
- Shows the preprocessing pipeline in action
- Useful for testing and understanding the preprocessing steps
- Includes examples of different data types and common data issues

Usage:
```bash
python dummy.py
```

### data_cleaning.py
Contains shared preprocessing functions and utilities used by both main.py and dummy.py.

## Input Data Format

The pipeline expects input data in CSV format with:
- Any number of numeric columns
- Any number of categorical columns
- Missing values (will be handled automatically)

## Output

The pipeline produces:
- A cleaned and preprocessed CSV file
- Detailed logs of the preprocessing steps
- Statistics about data transformations
- Information about missing values and outliers

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- scipy

## Installation

```bash
pip install pandas numpy scikit-learn scipy
```

## Tips for Usage

1. **Examining Data**
   - Check the initial data summary printed at load time
   - Review missing value counts
   - Monitor shape changes throughout processing

2. **Customization**
   - Adjust the outlier threshold in remove_outliers()
   - Modify the missing value strategy in handle_missing_values()
   - Add or remove preprocessing steps in the main pipeline

3. **Performance**
   - For large datasets, monitor memory usage
   - Consider batch processing for very large files

## Output Files

The pipeline creates:
- cleaned_preprocessed_data.csv (from main.py)
- preprocessed_dummy_data.csv (from dummy.py)

Each output file contains the fully processed dataset with:
- No missing values
- Scaled numeric features
- Encoded categorical variables
- Outliers removed

## Error Handling

The scripts include robust error handling for:
- File not found errors
- Invalid data formats
- Memory issues
- Type mismatches

All errors are reported with clear messages to help diagnose issues.