import pandas as pd
import os
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

df = pd.read_parquet(os.environ["DATASET_DIR"] + "fda_cleaned.parquet")

lb = LabelEncoder()

# --- Set top-K values ---
top_k_reactions = 100
top_k_drugs = 50

# --- Get top K reactions ---
top_reactions = df["reaction"].value_counts().nlargest(top_k_reactions)
top_reaction_values = top_reactions.index.tolist()

# Replace rare reactions with "other"
df["reaction"] = df["reaction"].where(df["reaction"].isin(top_reaction_values), "other")

# --- Get top K drugs ---
top_drugs = df["drug"].value_counts().nlargest(top_k_drugs)
top_drug_values = top_drugs.index.tolist()

# Replace rare drugs with "other"
df["drug"] = df["drug"].where(df["drug"].isin(top_drug_values), "other")

# Initialize LabelEncoders for each categorical column
label_encoders = {
    "country": LabelEncoder(),
    "reaction": LabelEncoder(),
    "drug": LabelEncoder(),
    "age_group": LabelEncoder(),
}

# Fit the LabelEncoders and transform the columns
encoded_mappings = {}
for col, le in label_encoders.items():
    # Fit and transform
    df[col] = le.fit_transform(df[col].astype("str"))

    # Create mapping dictionary
    encoded_mappings[col] = {
        "classes": le.classes_.tolist(),
        "mapping": {val: int(idx) for idx, val in enumerate(le.classes_)},
    }

df["reaction_outcome"] = lb.fit_transform(df["reaction_outcome"])

# Use stratify to maintain class distribution
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["reaction_outcome"], random_state=42
)

# For training data
train_df.reset_index(drop=True).to_parquet(
    os.environ["DATASET_DIR"] + "fd_train_df.parquet"
)

# For validation data
val_df.reset_index(drop=True).to_parquet(
    os.environ["DATASET_DIR"] + "fd_val_df.parquet"
)


# Save to JSON file
with open("mapper/category_mappings.json", "w") as f:
    json.dump(encoded_mappings, f, indent=2)

print("Successfully saved category mappings to 'category_mappings.json'")

# Create reverse mappings (encoded value → original string)
reverse_mappings = {}
for col, data in encoded_mappings.items():
    reverse_mappings[col] = {v: k for k, v in data["mapping"].items()}

# Save reverse mappings
with open("mapper/reverse_category_mappings.json", "w") as f:
    json.dump(reverse_mappings, f, indent=2)

print("Successfully saved reverse mappings to 'reverse_category_mappings.json'")
