import os
import zipfile
import ijson
import csv
import pandas as pd
from glob import glob
from dotenv import load_dotenv

load_dotenv()

ZIP_DIR = os.environ["OUTPUT_DIR"]
OUTPUT_CSV = os.environ["OUTPUT_CSV"]
OUTPUT_PARQUET = os.environ["OUTPUT_PARQUET"]

# Define column names for CSV
# csv_fields = [
#     "safetyreportid", "age", "age_unit", "sex", "country",
#     "reaction", "reaction_outcome", "drug", "drug_indication",
#     "startdate", "enddate"
# ]

csv_fields = ["age", "sex", "country", "reaction", "reaction_outcome", "drug"]

# Create and open CSV writer
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
    writer.writeheader()

    for zip_path in glob(f"{ZIP_DIR}/*.zip"):
        print(f"Processing {zip_path}...")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                for inner_file in z.namelist():
                    if inner_file.endswith(".json"):
                        with z.open(inner_file) as f:
                            # Stream JSON objects inside 'results'
                            try:
                                for report in ijson.items(f, "results.item"):
                                    patient = report.get("patient", {})
                                    drugs = patient.get("drug", [])
                                    reactions = patient.get("reaction", [])

                                    for reaction in reactions:
                                        for drug in drugs:
                                            row = {
                                                # "safetyreportid": report.get("safetyreportid"),
                                                "age": patient.get("patientonsetage"),
                                                # "age_unit": patient.get("patientonsetageunit"),
                                                "sex": patient.get("patientsex"),
                                                "country": report.get("occurcountry"),
                                                "reaction": reaction.get(
                                                    "reactionmeddrapt"
                                                ),
                                                "reaction_outcome": reaction.get(
                                                    "reactionoutcome"
                                                ),
                                                "drug": drug.get("medicinalproduct"),
                                                # "drug_indication": drug.get("drugindication"),
                                                # "startdate": drug.get("drugstartdate"),
                                                # "enddate": drug.get("drugenddate")
                                            }
                                            writer.writerow(row)
                            except Exception as e:
                                print(f"Error streaming {inner_file}: {e}")
        except Exception as e:
            print(f"Failed to open {zip_path}: {e}")

print(f"\nCompleted. Flattened data written to: {OUTPUT_CSV}")

pd.read_csv(OUTPUT_CSV).to_parquet(OUTPUT_PARQUET)
