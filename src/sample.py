import pandas as pd

df = pd.read_csv("data/processed/510k_2025.csv")

# "Unknown" has no specialty so can't be graded
df = df[df["advisory_committee_description"] != "Unknown"]

# 10 per specialty so every category gets a fair test
sample = df.groupby("advisory_committee_description").sample(n=10, random_state=7)

sample.to_csv("data/processed/eval_sample_holdout.csv", index=False)
print(len(sample))