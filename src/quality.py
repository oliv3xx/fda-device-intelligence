import re
import pandas as pd

SUFFIXES = {"inc", "llc", "ltd", "corp", "corporation", "co", "company", "gmbh", "sa", "ltda"}


def normalize(name):
    """Lowercase, strip punctuation and legal suffixes."""
    name = name.lower()
    name = re.sub(r"[^\w\s]", "", name)
    words = [w for w in name.split() if w not in SUFFIXES]
    return " ".join(words)


df = pd.read_csv("data/processed/510k_2025.csv")
print("Total records:", len(df))

# missing values
print("\nMissing values per column:")
print(df.isna().sum())

# duplicate IDs
dupes = df["k_number"].duplicated().sum()
print("\nDuplicate k_numbers:", dupes)

# dates outside 2025
df["decision_date"] = pd.to_datetime(df["decision_date"])
bad_dates = (df["decision_date"].dt.year != 2025).sum()
print("Dates outside 2025:", bad_dates)

# same company often appears with different suffixes/punctuation,
# so normalize names before counting
df["applicant_clean"] = df["applicant"].apply(normalize)
raw = df["applicant"].nunique()
clean = df["applicant_clean"].nunique()
print(f"\nCompany names: {raw} raw -> {clean} after cleaning ({raw - clean} duplicates)")

# spot check merges to make sure different companies weren't combined
merged = df.groupby("applicant_clean")["applicant"].unique()
merged = merged[merged.apply(len) > 1]
print("\nSample merged names:")
print(merged.head(15))