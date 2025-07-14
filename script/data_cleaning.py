import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder

load_dotenv()

OUTPUT_PARQUET = os.environ["OUTPUT_PARQUET"]
df = pd.read_parquet(OUTPUT_PARQUET)

# 1. Count missing values in age
total_rows = len(df)
null_age_count = df["age"].isnull().sum()
print(
    f"Missing age: {null_age_count} out of {total_rows} ({null_age_count / total_rows:.2%})"
)

# 2. Create age_group column
conditions = [
    (df["age"] < 11),
    (df["age"].between(11, 20)),
    (df["age"].between(21, 30)),
    (df["age"].between(31, 40)),
    (df["age"].between(41, 50)),
    (df["age"].between(51, 60)),
    (df["age"].between(61, 70)),
    (df["age"].between(71, 80)),
    (df["age"] > 80),
]

choices = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81+"]

df["age_group"] = np.select(conditions, choices, default="Unknown")

# 3. Calculate mean age
mean_age = df["age"].mean()

# 4. Filter out null reaction_outcome
df = df[df["reaction_outcome"].notna()]

# 5. Fill missing values
df = df.fillna({"country": "unknown", "sex": 0.0, "age": mean_age})

# 6. Identify numeric and non-numeric columns
numeric_cols = df.select_dtypes(include=["number"]).columns
all_cols = df.columns

# 7. Calculate missing values per column
missing_values = pd.DataFrame(
    {
        "column": all_cols,
        "missing_count": [
            (
                df[col].isnull().sum()
                if col not in numeric_cols
                else (df[col].isnull() | np.isnan(df[col])).sum()
            )
            for col in all_cols
        ],
    }
)

print("\nMissing values per column:")
print(missing_values)

df.to_parquet("data/fda_cleaned.parquet", engine="pyarrow")

# with open('mapper/category_mappings.json') as f:
#     category_mappings = json.load(f)


# categorical_cols = ['country', 'reaction', 'drug', 'age_group']

# # 4. Apply the mappings to each column
# for col in categorical_cols:
#     # Create mapping dictionary
#     mapping = category_mappings[col]['mapping']

#     # Map the values (using .get() with a default for unseen values)
#     df[col] = df[col].map(lambda x: mapping.get(str(x), -1))  # -1 for unknown values

#     # Convert to appropriate dtype (smallest possible unsigned int)
#     max_val = df[col].max()
#     if max_val < 255:
#         df[col] = df[col].astype('uint8')
#     elif max_val < 65535:
#         df[col] = df[col].astype('uint16')
#     else:
#         df[col] = df[col].astype('uint32')

# print("\nEncoded value counts:")
# for col in categorical_cols:
#     print(f"\n{col}:")
#     print(df[col].value_counts().head())

# df.to_parquet("data/fda_cleaned.parquet", engine='pyarrow')
# print("\nEncoded DataFrame saved to 'fda_cleaned.parquet'")

# print(df)
