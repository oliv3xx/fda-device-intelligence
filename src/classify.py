import json
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

MODEL = "gemini-3.5-flash-lite"
BATCH_SIZE = 20

VERSION = "v3"

GUIDANCE = """Notes on FDA categories (they follow FDA regulation, not just the body part):
- Nerve stimulators (TENS), brain/spine navigation, pain or cognitive assessment tools -> Neurology
- Lab tests go by what they measure: antibodies/immune markers -> Immunology; chemicals, drug levels, proteins in blood -> Clinical Chemistry; infections/pathogens -> Microbiology; blood cells, clotting, semen analysis -> Hematology; drugs of abuse -> Clinical Toxicology
- Diabetes devices (insulin pumps, glucose monitors) -> Clinical Chemistry
- Digital pathology viewers and slide software -> Pathology
- Breathing devices (nebulizers, oxygen concentrators, CPAP masks, tracheostomy tubes) -> Anesthesiology
- General hospital supplies (gloves, syringes, sterilization, trays) -> General Hospital
- Bronchoscopes, airway endoscopes, laryngeal tubes, ear implants and hearing devices -> Ear, Nose, Throat (even though they relate to breathing)
"""

sample = pd.read_csv("data/processed/510k_2025_sample.csv")

# sample is grouped by specialty; shuffle so each batch is mixed
sample = sample.sample(frac=1, random_state=42).reset_index(drop=True)
categories = sorted(sample["advisory_committee_description"].unique())

def build_prompt(device_names):
    category_list = "\n".join(f"- {c}" for c in categories)
    device_list = "\n".join(f"{i}. {name}" for i, name in enumerate(device_names))
    guidance = GUIDANCE if VERSION in ("v2", "v3") else ""
    return f"""Classify each FDA medical device into exactly one medical specialty from this list:
{category_list}

{guidance}
Devices:
{device_list}

Return JSON: a list of objects with "id" (the device number) and "specialty" (copied exactly from the list)."""

predictions = []
for start in range(0, len(sample), BATCH_SIZE):
    batch = sample.iloc[start:start + BATCH_SIZE]
    prompt = build_prompt(batch["device_name"].tolist())

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    results = json.loads(response.text)

    # match answers back by id in case the model skips/reorders
    lookup = {int(r["id"]): r["specialty"] for r in results}
    for i in range(len(batch)):
        predictions.append(lookup.get(i))

    print(f"Done {start + len(batch)} / {len(sample)}")
    time.sleep(5)  # stay under 15 requests/minute

sample["predicted"] = predictions
sample.to_csv(f"data/processed/predictions_{VERSION}.csv", index=False)