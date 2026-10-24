# GlucoWise: synthetic CGM data, glucose analysis and AI hand-off

Synthetic CGM and meal data for scenarios A, B and C (3 users, 7 days, 21 meals each, 2026-10-01 to 2026-10-07),
Python functions that compute post-meal glucose metrics and compare past meals, and helpers that pass the results
to the Microsoft Foundry agent. **All glucose and meal values are invented for a hackathon demo.** They are not real
measurements, not medically validated, and not evidence that any food causes a specific glucose response.

## Files

| Path | Purpose |
| --- | --- |
| `data/cgm.csv` | 6,048 CGM readings, every 5 minutes, users A, B, C |
| `data/meals.csv` | 63 meals, one row per meal |
| `src/data_generator.py` | Regenerates both CSVs (fixed seed, DST-safe, LF line endings) |
| `src/schema.py` | Validates the CSVs (columns, IDs, timestamps, 5-minute grid, user/meal linkage) |
| `src/glucose_analysis.py` | Baseline, peak and rise per meal; meal history; food comparison; per-user patterns |
| `src/agent_context.py` | Builds the text (analysis JSON + question) sent to the Foundry agent |
| `src/pubmed_search.py` | PubMed search (NCBI E-utilities); returns real records, never invents papers |
| `tests/` | `unittest` tests: data and validator, analysis, agent context, PubMed parsing (offline) |
| `scripts/inspect_existing_csvs.py` | Read-only check of older CSVs against the new schema |
| `docs/data_schema.md` | Column definitions |
| `docs/agent_integration.md` | Agent instructions, Streamlit snippets, accuracy checks for the AI answers |

## Run (from the project root)

```bash
python -m pip install pandas numpy
python src/data_generator.py     # rewrites data/cgm.csv and data/meals.csv
python src/schema.py             # validates them
python src/glucose_analysis.py   # prints the per-user food summary (JSON)
python -W error -m unittest discover -s tests     # all tests, warnings fail
python src/pubmed_search.py "sulforaphane glucose metabolism"   # needs internet
```

Only `pandas` and `numpy` are needed (`tzdata` comes with pandas). The Foundry connection additionally needs
`azure-ai-projects` and `azure-identity`, and an Azure login (see `docs/agent_integration.md`).

## How the analysis works

- **baseline** = mean of the CGM readings in the 15 minutes before the meal
- **peak** = highest reading from the meal time up to 2 hours after it
- **rise** = peak - baseline (mg/dL)

```python
from src.glucose_analysis import get_meal_history, get_meal_glucose, get_glucose_curve, compare_foods, analyze_user_patterns

meal_id = get_meal_history("A")[0]["meal_id"]   # ids look like A_D01_1_white_rice
get_meal_glucose("A", meal_id)                  # {"baseline", "peak", "rise", "unit", "data_type": "simulated"}
get_glucose_curve("A", meal_id)                 # -30 to +120 min, for charts
compare_foods("A")                              # per-food meal_count and average rise
analyze_user_patterns("A")                      # summary handed to the AI
```

Food names are the CSV keys (`white_rice`, `whole_oats`, ...). Foods are compared within one user only.

## Status and testing

- All tests pass on Linux (Python 3.13, pandas 3.0.5, `-W error`) and on Windows (61 tests at the time of testing; Python/pandas versions there were not recorded).
- The reference averages for user A (white rice 73.9, brown rice 41.0, whole oats 31.3 mg/dL) are checked by a test.
- PubMed search was run live on one query and returned real records; offline tests cover parsing and error handling.
- The Foundry agent calls and the Streamlit integration are described in `docs/agent_integration.md` but are not covered by the tests here.

## Limitations

- **Time to peak is synthetic.** It comes from a schedule-position cycle (40-60 min) that is the same for every user. Do not treat it as a food or person effect.
- Pre-meal readings are exactly flat (no noise), unlike real sensors.
- Only the primary food's grams are stored (`primary_portion_g`); side dishes appear only in `description`, so comparisons are by primary food. Side dishes and meal slots vary between meals of the same food, and some foods have only 3 repeats.
- Scenario D (meal order) is intentionally not included.
- PubMed hits only match search words. They are not verified evidence for a user's question, and many are cell or animal studies. The AI must check relevance and study type (see `docs/agent_integration.md`).

## Schema notes

- `primary_portion_g` is grams of the primary comparison food only; `carbs_g` is blank (unknown).
- The oats food key is `whole_oats`.
- Timestamps are ISO-8601 with `America/Los_Angeles` UTC offsets.
- These CSVs replace the older 864-row demo CSV (no `user_id` column). `requirements.txt` still lacks `streamlit` and the Azure packages.