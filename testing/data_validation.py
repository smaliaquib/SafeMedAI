import pandas as pd
from typing import Dict, List, Optional

def validate_pharma_dataset(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Comprehensive validation for the pharmaceutical dataset.
    Returns a dictionary with validation results.
    """
    validation_results = {}
    
    # 1. Schema Validation
    expected_schema = {
        'age': 'float64',
        'sex': 'float64',
        'country': 'uint8',
        'reaction': 'uint8',
        'reaction_outcome': 'int64',
        'drug': 'uint8',
        'age_group': 'uint8'
    }
    validation_results['schema_validation'] = validate_schema(df, expected_schema)
    
    # 2. Missing Values Validation
    validation_results['missing_values'] = validate_no_missing_values(df)
    
    # 3. Binary Column Validation (assuming sex should be 0, 1, or 2)
    # Note: Your data shows 0,1,2 - adjust if needed
    validation_results['sex_validation'] = validate_categorical_column(df, 'sex', [0.0, 1.0, 2.0])
    
    # 4. Age Validation (non-negative, reasonable range)
    validation_results['age_validation'] = validate_numeric_range(df, 'age', min_val=0, max_val=120)
    
    # 5. Reaction Outcome Validation (assuming it's categorical 1-5)
    validation_results['reaction_outcome_validation'] = validate_categorical_column(
        df, 'reaction', [0, 1, 2, 3, 4, 5])
    
    # 6. Age Group Validation (assuming it's categorical based on your data)
    validation_results['age_group_validation'] = validate_numeric_range(df, 'age_group', min_val=0, max_val=9)
    
    # 7. Non-negative values for all numeric columns
    validation_results['non_negative_values'] = validate_non_negative_values(df)
    
    return validation_results

# Helper validation functions (similar to previous but with some additions)

def validate_schema(df: pd.DataFrame, expected_schema: Dict[str, type]) -> bool:
    """Validates that the dataframe matches an expected schema."""
    validation_passed = True
    
    # Check if all expected columns exist
    missing_columns = set(expected_schema.keys()) - set(df.columns)
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        validation_passed = False
    
    # Check if any extra columns exist
    extra_columns = set(df.columns) - set(expected_schema.keys())
    if extra_columns:
        print(f"Extra columns: {extra_columns}")
        validation_passed = False
    
    # Check column types
    for column, expected_type in expected_schema.items():
        if column in df.columns:
            actual_type = df[column].dtype
            if not pd.api.types.is_dtype_equal(actual_type, expected_type):
                print(f"Column '{column}' has incorrect type. Expected: {expected_type}, Actual: {actual_type}")
                validation_passed = False
    
    if validation_passed:
        print("Schema validation passed successfully!")
    return validation_passed

def validate_no_missing_values(df: pd.DataFrame, columns: Optional[List[str]] = None) -> bool:
    """Checks for and reports any missing (null) values in the dataframe."""
    if columns is None:
        columns = df.columns
    
    validation_passed = True
    for column in columns:
        if column in df.columns:
            null_count = df[column].isnull().sum()
            if null_count > 0:
                print(f"Column '{column}' has {null_count} missing values")
                validation_passed = False
    
    if validation_passed:
        print("No missing values found in the specified columns!")
    return validation_passed

def validate_categorical_column(df: pd.DataFrame, column: str, allowed_values: List) -> bool:
    """Validates that a column contains only specified categorical values."""
    if column not in df.columns:
        print(f"Column '{column}' not found in dataframe")
        return False
    
    unique_values = set(df[column].dropna().unique())
    invalid_values = unique_values - set(allowed_values)
    
    if invalid_values:
        print(f"Column '{column}' contains invalid values: {invalid_values}")
        print(f"Allowed values are: {allowed_values}")
        return False
    
    print(f"Column '{column}' contains only valid categorical values")
    return True

def validate_numeric_range(df: pd.DataFrame, column: str, min_val: float = None, max_val: float = None) -> bool:
    """Validates that a numeric column falls within specified range."""
    if column not in df.columns:
        print(f"Column '{column}' not found in dataframe")
        return False
    
    validation_passed = True
    
    if min_val is not None:
        below_min = (df[column] < min_val).sum()
        if below_min > 0:
            print(f"Column '{column}' has {below_min} values below minimum {min_val}")
            validation_passed = False
    
    if max_val is not None:
        above_max = (df[column] > max_val).sum()
        if above_max > 0:
            print(f"Column '{column}' has {above_max} values above maximum {max_val}")
            validation_passed = False
    
    if validation_passed:
        range_msg = []
        if min_val is not None:
            range_msg.append(f"≥{min_val}")
        if max_val is not None:
            range_msg.append(f"≤{max_val}")
        print(f"Column '{column}' values are within expected range ({' '.join(range_msg)})")
    
    return validation_passed

def validate_non_negative_values(df: pd.DataFrame, columns: Optional[List[str]] = None) -> bool:
    """Validates that specified columns contain only non-negative values."""
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    
    validation_passed = True
    for column in columns:
        if column in df.columns:
            negative_count = (df[column] < 0).sum()
            if negative_count > 0:
                print(f"Column '{column}' has {negative_count} negative values")
                validation_passed = False
    
    if validation_passed:
        print("All numeric columns contain only non-negative values!")
    return validation_passed

# Example usage with your dataset
if __name__ == "__main__":
    df = pd.read_parquet("data/fda_cleaned.parquet")
    print(df['reaction'].unique())
    # Load your dataset (replace this with your actual data loading code)    
    # Run validations
    validation_results = validate_pharma_dataset(df)
    
    print("\n=== Validation Summary ===")
    for test_name, result in validation_results.items():
        print(f"{test_name}: {'PASS' if result else 'FAIL'}")
    
    # Check if all validations passed
    if all(validation_results.values()):
        print("\nAll validations passed successfully!")
    else:
        print("\nSome validations failed. Please check the output for details.")