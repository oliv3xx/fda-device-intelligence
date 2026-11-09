import json
import os
import pandas as pd

# load raw json
with open("data/raw/510k_2025.json") as f:
    records = json.load(f)

# flatten nested fields into columns
df = pd.json_normalize(records)

# keep columns needed
columns = [
    "k_number",
    "applicant",
    "device_name",
    "product_code",
    "decision_date",
    "decision_description",
    "advisory_committee_description",
    "openfda.device_class",
]
df = df[columns]

# FDA uses empty strings for some missing values; treat them as blanks
df = df.replace("", pd.NA)

df = df.rename(columns={"openfda.device_class":"device_class"})

# save to csv
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/510k_2025.csv", index=False)

print(df.shape)
print(df.head())

print(df.isna().sum())