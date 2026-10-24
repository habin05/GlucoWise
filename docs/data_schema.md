# GlucoWise CSV schema v1.1 (Step 1 revision)

All data is **synthetic**. It is not real CGM data, not medically validated, and
must not be presented as a medical finding.

Both files are plain UTF-8 CSV with a header row. Read every cell as text first
(`pd.read_csv(..., dtype=str, keep_default_na=False)` or `src/schema.py`'s
`read_csv_as_text`), so a user ID such as `NA` is never turned into a missing value.

## `data/cgm.csv`
One record per user per timestamp. Sorted by `user_id`, then time; each user's rows are contiguous.

| Column | Required | Meaning |
|---|---|---|
| `user_id` | yes | Synthetic user key, `[A-Z][A-Z0-9]{0,15}` (e.g. `A`, `NA`, `U001`) |
| `timestamp` | yes | `YYYY-MM-DDTHH:MM:SS±HH:MM`; offset must be the correct `America/Los_Angeles` offset at that instant |
| `glucose_mg_dl` | yes | Synthetic glucose value in mg/dL |

Rules: `(user_id, timestamp)` unique; exactly 300 s between consecutive readings per user;
readings on :00/:05/:10… minutes. A day with a daylight-saving change has 23 or 25 hours of readings.

## `data/meals.csv`
One record per meal. Sorted by `user_id`, then time.

| Column | Required | Meaning |
|---|---|---|
| `meal_id` | yes | Unique, `USER_Ddd_slot_food`, e.g. `A_D01_1_white_rice`; prefix must equal `user_id`, suffix must equal `food` |
| `user_id` | yes | Must exist in `cgm.csv` |
| `scenario_id` | yes | `A`, `B` or `C` (Scenario D is not implemented); one scenario per user |
| `timestamp` | yes | Meal start, same format as above |
| `food` | yes | Lowercase canonical key of the **primary comparison food** (`white_rice`, `whole_oats`, `banana`…) |
| `description` | yes | Human-readable full meal composition and amounts |
| `primary_portion_g` | optional | Grams of the **primary comparison food only**, not the whole meal. Renamed from `portion_g` |
| `carbs_g` | optional | Carbohydrate grams for the **whole meal** when known. **Blank means unknown**; never treat blank as 0 |
| `meal_order` | yes | `unspecified` for now; the allowed list will grow when Scenario D is defined |

There is deliberately **no** `carbs_g <= primary_portion_g` rule, because the two columns describe different things.

## Synthetic time-to-peak: do not interpret

The generator places each peak at a minute taken from `PEAK_MINUTES`, a fixed 21-step cycle (40–60 min)
applied by schedule position and identical for every user. **Time to peak is therefore a drawing
choice of the simulator, not a food or person effect.** Do not compare it across foods or users, and do not
let the AI agent explain it. Baseline, peak and rise are the scenario's designed targets.

Each meal's window has exactly one maximum reading: all other readings from meal time to +120 min are
capped 0.2 mg/dL below the target peak.

## Metric definitions used for validation (Step 2 will reuse them)

- baseline = mean of the readings from 15 min before the meal through the meal time (4 readings)
- peak = maximum reading after the meal through +120 min (24 readings)
- rise = peak − baseline
