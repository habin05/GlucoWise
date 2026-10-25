# GlucoWise

**Know Your Glucose. Live Wiser.**

GlucoWise is a Streamlit web app that connects meal photos with CGM (continuous glucose monitor) data, shows how glucose changed after each meal, and lets a Microsoft Foundry AI agent explain the results in plain language.

> **Demo only.** All glucose and meal values in this repo are invented for a hackathon demo. They are not real measurements, have not been medically validated, and are not evidence that any food causes a specific glucose response. GlucoWise does not provide medical advice.

## Features

The app requires login and has four tabs.

| Tab | What it does |
| --- | --- |
| **1. Data Input** | Upload a CGM CSV and save it. Upload several food photos at once, recognize the foods with AI, set each meal's date and time, and save the records. Food names are normalized on save. Saved meals can be deleted, and a reset button clears your uploaded data. |
| **2. Meal Analysis** | Pick a date to see its CGM chart with meal markers, a per-meal table (baseline, peak, rise, notes), and a detail view with the glucose curve around the selected meal. |
| **3. Historical Patterns** | Compares foods within your own records: meal count and average rise per food. Foods with fewer than 3 meals are flagged as a small sample. Meals without enough CGM data are excluded and listed with a warning. |
| **4. AI Insights** | Chat with the Foundry agent about your results, or use the one-click automatic summary. The last 8 messages are sent as context, and the conversation resets when the user changes. Include the word "pubmed" in a question to attach PubMed search results to the prompt. |

The AI only explains numbers that Python has already calculated. It is instructed not to invent glucose values or claim that a food caused a rise.

## Quick start

Run everything from the project root.

```bash
git clone https://github.com/habin05/GlucoWise.git
cd GlucoWise
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

**1. Create a login account** (once per account). Each account maps to one synthetic CGM user, A to F. Passwords are stored only as salted PBKDF2 hashes.

```bash
python src/auth_db.py add <username> A     # asks for a password
python src/auth_db.py list
```

**2. Configure the AI features** (see the next section), then start the app:

```bash
streamlit run src/app.py
```

### AI features (Azure)

Food recognition and the AI chat call Microsoft Foundry agents. They need an Azure login that `DefaultAzureCredential` can use (for example `az login`) and these environment variables:

| Variable | Purpose |
| --- | --- |
| `FOUNDRY_ENDPOINT` | Foundry project endpoint (optional; the code has a default) |
| `FOUNDRY_AGENT` | Chat agent name (default: `GlucoWise-Agent`) |
| `FOOD_NORMALIZER_AGENT` | Food normalization agent name (default: `GlucoWise-Food-Normalizer`) |

PowerShell example:

```powershell
$env:FOUNDRY_ENDPOINT = "https://<your-resource>.services.ai.azure.com/api/projects/<your-project>"
streamlit run src/app.py
```

Optional for PubMed search: `NCBI_API_KEY` (higher rate limit) and `NCBI_EMAIL` (contact address for NCBI).

Without Azure, login, CGM upload, meal entry and all analysis tabs still work; only AI recognition and AI chat fail. Agent instructions and accuracy checks are in [`docs/agent_integration.md`](docs/agent_integration.md).

## How the analysis works

For each meal:

- **baseline** = mean of the CGM readings in the 15 minutes before the meal
- **peak** = highest reading from the meal time up to 2 hours after it
- **rise** = peak − baseline (mg/dL)

A meal is excluded from the analysis when there is not enough CGM data around it. Rise is calculated once per meal and is not split across the foods in that meal. Glucose values always come directly from the uploaded or sample CSV; no values are generated at run time.

```python
from src.glucose_analysis import (
    get_meal_history, get_meal_glucose, get_glucose_curve,
    compare_foods, analyze_user_patterns,
)

meal_id = get_meal_history("A")[0]["meal_id"]   # ids look like A_D01_1_white_rice
get_meal_glucose("A", meal_id)                  # {"baseline", "peak", "rise", "unit", "data_type": "simulated"}
get_glucose_curve("A", meal_id)                 # -30 to +120 min, for charts
compare_foods("A")                              # per-food meal_count and average rise
analyze_user_patterns("A")                      # summary handed to the AI
```

Foods are compared **within one user only**.

## Project structure

| Path | Purpose |
| --- | --- |
| `src/app.py` | Streamlit app (four tabs) |
| `src/auth_db.py` | Local demo login (SQLite, salted PBKDF2 hashes) and account CLI |
| `src/user_data.py` | Per-user CGM and meal storage, CGM CSV parsing, meal status checks, data reset |
| `src/local_db.py` | Saved meal records in `data/glucowise.db` |
| `src/food_recognition.py` | AI food recognition from photos and text normalization (Foundry) |
| `src/normalization.py` | Food name normalization |
| `src/glucose_analysis.py` | Baseline, peak and rise per meal; meal history; food comparison; per-user patterns |
| `src/agent_context.py` | Builds the text (analysis JSON + question) sent to the Foundry agent |
| `src/pubmed_search.py` | PubMed search via NCBI E-utilities; returns real records, never invents papers |
| `src/schema.py` | Validates the CSVs (columns, IDs, timestamps, 5-minute grid, user/meal linkage) |
| `src/data_generator.py` | Regenerates the demo CSVs (fixed seed, DST-safe, LF line endings) |
| `src/make_demo_cgm.py` | Creates a one-day demo CGM CSV you can upload in the app |
| `data/cgm.csv`, `data/meals.csv` | Demo data: 6,048 CGM readings and 63 meals for users A, B, C |
| `tests/` | `unittest` tests |
| `scripts/inspect_existing_csvs.py` | Read-only check of older CSVs against the new schema |
| `docs/data_schema.md` | Column definitions |
| `docs/agent_integration.md` | Agent instructions and accuracy checks for AI answers |

## Demo data

Scenarios A, B and C: 3 users, 7 days, 21 meals each (2026-10-01 to 2026-10-07).

```bash
python src/data_generator.py     # regenerates data/cgm.csv and data/meals.csv
python src/schema.py             # validates them
python src/glucose_analysis.py   # prints the per-user food summary (JSON)
python src/make_demo_cgm.py --date 2026-10-24 --meals 08:00:72,12:30:33 --out demo_cgm.csv   # a CSV to upload in the app
python src/pubmed_search.py "sulforaphane glucose metabolism"   # needs internet
```

## Testing

```bash
python -W error -m unittest discover -s tests    # warnings fail the run
```

All 142 tests pass. They cover data generation and validation, glucose analysis, CGM parsing, login, agent context building, and PubMed parsing (offline). The Foundry calls and the Streamlit screens are **not** covered by automated tests.

## Limitations

- **Time to peak is synthetic.** It comes from a schedule-position cycle (40–60 min) that is the same for every user. Do not treat it as a food or person effect.
- Pre-meal readings in the demo data are exactly flat (no noise), unlike real sensors.
- Only the primary food's grams are stored (`primary_portion_g`); side dishes appear only in `description`, so comparisons are by primary food. Some foods have only 3 repeats.
- Meal size, side dishes, exercise and sleep also affect glucose. The Historical Patterns tab shows observed patterns in the records, not causes.
- The in-app PubMed lookup currently runs one fixed query (`sulforaphane glucose metabolism`) when a question mentions "pubmed". Results only match search words; they are not verified evidence for a user's question, and many are cell or animal studies. The AI must check relevance and study type (see `docs/agent_integration.md`).
- The login is a local demo feature and is not meant for deployment.
- Scenario D (meal order) is intentionally not included.

## Schema notes

- `primary_portion_g` is the grams of the primary comparison food only; `carbs_g` is blank (unknown).
- The oats food key is `whole_oats`.
- Timestamps are ISO-8601 with `America/Los_Angeles` UTC offsets.
- Uploaded CGM files with no time zone are interpreted in `America/Los_Angeles`.