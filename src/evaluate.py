import pandas as pd

df = pd.read_csv("data/processed/predictions.csv")

# true when Gemini matched FDA's specialty
df["correct"] = df["predicted"] == df["advisory_committee_description"]

# overall accuracy
accuracy = df["correct"].mean()
print(f"Overall accuracy: {accuracy:.1%} ({df['correct'].sum()}/{len(df)})")

# accuracy per specialty, worst first
by_specialty = df.groupby("advisory_committee_description")["correct"].mean().sort_values()
print("\nAccuracy by specialty:")
print(by_specialty.apply(lambda x: f"{x:.0%}"))

# what Gemini guessed when it was wrong
mistakes = df[~df["correct"]]
common_errors = (
    mistakes.groupby(["advisory_committee_description", "predicted"])
    .size()
    .sort_values(ascending=False)
)
print("\nMost common mistakes (actual -> predicted):")
print(common_errors.head(10))

by_specialty.to_csv("data/processed/accuracy_by_specialty.csv")