import pandas as pd

VERSIONS = ["v1_holdout", "v3_holdout"]
by_specialty = {}

for version in VERSIONS:
    df = pd.read_csv(f"data/processed/predictions_{version}.csv")
    df["correct"] = df["predicted"] == df["advisory_committee_description"]
    print(f"{version} accuracy: {df['correct'].mean():.1%} ({df['correct'].sum()} / {len(df)})")
    by_specialty[version] = df.groupby("advisory_committee_description")["correct"].mean()

comparison = pd.DataFrame(by_specialty)

# Compare the newest version to the one before it
latest, previous = VERSIONS[-1], VERSIONS[-2]
comparison["change"] = comparison[latest] - comparison[previous]
comparison = comparison.sort_values("change")

print(f"\nAccuracy by specialty (change = {latest} vs {previous}):")
print((comparison * 100).round(0))

comparison.to_csv("data/processed/accuracy_comparison.csv")