import requests
import json

URL = "https://api.fda.gov/device/510k.json"
SEARCH = "decision_date:[20250101 TO 20251231]"
LIMIT = 1000  # max records openFDA returns per request

all_results = []
skip = 0

while True:
    params = {"search": SEARCH, "limit": LIMIT, "skip": skip}
    response = requests.get(URL, params=params)
    data = response.json()

    # add records to all_results
    all_results.extend(data["results"])

    # move skip forward 
    skip += LIMIT

    # break once everything collected
    if skip >= data["meta"]["results"]["total"]:
        break

# save results to a JSON file
with open("data/raw/510k_2025.json", "w") as f:
    json.dump(all_results, f)

print(len(all_results))
