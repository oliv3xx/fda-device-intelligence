# FDA Device Intelligence

A data pipeline that pulls every FDA 510(k) medical device clearance from 2025, measures how clean the data is, and tests whether an LLM can classify devices into medical specialties from the device name alone.

Data pulled from the openFDA API in November 2026. The FDA updates this database over time, so rerunning the fetch script later may give slightly different counts.

## Why this project

Medtech companies use FDA clearance data to track competitors and spot new products. To be useful, that data has to be clean (one company should count as one company) and well categorized. This project looks at both problems:

1. How messy is public FDA device data, and what does it take to clean it?
2. Can an LLM reliably tag devices by specialty, and how do you measure that?

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1. Acquire | `src/fetch.py` | Pulls all 2025 510(k) records from the openFDA API, paging 1,000 at a time |
| 2. Flatten | `src/flatten.py` | Converts nested JSON into a clean 8-column CSV |
| 3. Data quality | `src/quality.py` | Checks completeness, duplicates, date ranges, and company name variants |
| 4. Sample | `src/sample.py` | Builds an eval set of 10 devices per specialty |
| 5. Classify | `src/classify.py` | Sends device names to Gemini in batches of 20 and saves its predicted specialty |
| 6. Evaluate | `src/evaluate.py` | Scores accuracy overall and per specialty, and compares prompt versions |

## Data quality results

3,225 device clearances from 2025.

| Check | Result |
|---|---|
| Duplicate IDs | 0 |
| Decision dates outside 2025 | 0 |
| Missing device class | 3 |
| Missing product code | 2 |
| Specialty listed as "Unknown" | 63 (2%) |
| Company name variants | 2,036 raw names, 1,993 after cleaning (43 duplicates) |

Things I found along the way:

- **Missing values hidden as empty strings.** My missing-value counts didn't match between two steps. The FDA stores some missing product codes as `""` instead of null, so a basic null check missed them. I fixed the pipeline to treat both as missing.
- **Missing values hidden as text.** 63 records list their specialty as "Unknown," which a null check also doesn't catch. I left these out of the eval since there's no right answer to grade against.
- **Same company, different spellings.** "Edwards Lifesciences" vs "Edwards Lifesciences, LLC," "Cepheid®" vs "Cepheid." I normalized names (case, punctuation, legal suffixes) and spot checked the merged groups to make sure different companies weren't combined. One edge case: bioMerieux's US branch and French parent were merged, which is a judgment call.

## LLM classification eval

**Task:** given only a device name, pick its FDA specialty from 19 options. The FDA's own `advisory_committee_description` field is the answer key.

**Setup**
- Model: Gemini 3.5 Flash Lite (free tier)
- Eval set: 190 devices, 10 per specialty, so small categories get a fair test
- Fixed random seed, so every prompt version is tested on the same devices

**Prompt versions**
- **v1:** list of specialties plus the device names
- **v2:** added short guidance on categories the model kept confusing (lab tests, nerve stimulators, breathing devices, pathology software)
- **v3:** added an exception for airway scopes and ear devices, after v2 pulled bronchoscopes out of ENT

| Version | Tuning sample | Held-out sample |
|---|---|---|
| v1 | 64.7% | 68.9% |
| v2 | 72.1% | |
| v3 | 74.2% | 72.6% |

Random guessing across 19 specialties would score about 5%.

### What the eval showed

- **Part of the gain was overfitting.** On the sample I tuned against, v3 beat v1 by about 10 points. On a fresh sample of 190 devices it was about 4 points. The rules were partly fit to the specific devices I'd looked at.
- **The targeted fixes did hold up.** On the held-out sample, Neurology and Anesthesiology each improved 30 points, and Pathology and Hematology each improved 20.
- **Fixing one category can break another.** v2's "breathing devices → Anesthesiology" rule dropped ENT from 70% to 40% by pulling in bronchoscopes. v3 fixed it.
- **Many "errors" are really about how the FDA organizes devices.** TENS units are filed under Neurology, digital pathology viewers under Pathology, and semen analyzers under Hematology. The model's guesses were often reasonable, just not how the FDA categorizes them.
- **The model isn't always consistent.** The same device name (microINR System) appeared twice and got two different answers.

### Limitations

- 10 devices per specialty means a 10-point change is a single device, so I only treated big swings as real.
- Radiology (30 to 60%) and Immunology (20 to 50%) stayed weak across every version. Imaging software and lab tests are hard to tell apart from the name alone.
- A natural next step would be giving the model more than the name, like the product code description or the 510(k) summary.

## How to run

```
python -m venv venv
venv\Scripts\activate
pip install requests pandas google-genai python-dotenv
```

Create a `.env` file in the project folder with a free Gemini API key from Google AI Studio:

```
GEMINI_API_KEY=your-key-here
```

Then run the scripts in order from the project folder:

```
python src/fetch.py
python src/flatten.py
python src/quality.py
python src/sample.py
python src/classify.py
python src/evaluate.py
```

`classify.py` has `VERSION` and `SAMPLE_FILE` settings at the top for choosing the prompt version and eval set.

## Project structure

```
data/
  raw/          # raw API data (not committed, rebuilt by fetch.py)
  processed/    # cleaned CSV, eval samples, predictions, results
src/            # pipeline scripts
```