import os
import json

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

from local_db import init_db, save_meal, get_meals, delete_meal

from glucose_analysis import (
    get_meal_history,
    get_meal_glucose,
    get_glucose_curve,
    compare_foods,
    analyze_user_patterns,
)
from agent_context import build_agent_prompt
from auth_db import login_gate
import upload_store
from cgm_import import parse_cgm_csv, CgmImportError, DEFAULT_TZ
from meal_analysis import summarize_by_food

# Test switch: run with  GLUCOWISE_MOCK=1  to skip Azure and return a fake answer.
# Off by default, so a normal run (or a push) can never ship mock answers.
USE_MOCK = os.environ.get("GLUCOWISE_MOCK") == "1"

st.set_page_config(
    page_title="GlucoWise",
    page_icon="📈",
    layout="wide",
)

user_id = login_gate()

st.title("📈 GlucoWise")
st.caption("Understand how meals affect your glucose levels")

st.info(
    "Demo mode: All glucose data shown here are simulated. "
    "This app does not provide medical advice."
)

if USE_MOCK:
    st.warning("Mock mode is ON: the AI answers below are fake (no Azure call).")

# Initialize local SQLite database
init_db()
upload_store.init_db()

# Add a new meal
st.header("➕ Add a meal")

with st.form("add_meal_form"):
    meal_date = st.date_input("Meal date")
    meal_time = st.time_input("Meal time")
    food = st.text_input("Food name", placeholder="e.g. Chicken salad")
    description = st.text_input("Description", placeholder="e.g. Lunch")
    portion_g = st.number_input(
        "Portion (g)", min_value=0.0, value=100.0, step=10.0
    )
    carbs_g = st.number_input(
        "Carbohydrates (g)", min_value=0.0, value=0.0, step=5.0
    )

    submitted = st.form_submit_button("Save meal")

    if submitted:
        if not food.strip():
            st.error("Please enter a food name.")
        else:
            timestamp = datetime.combine(
                meal_date, meal_time
            ).isoformat(timespec="minutes")

            save_meal(
                timestamp=timestamp,
                food=food.strip(),
                description=description.strip(),
                portion_g=portion_g,
                carbs_g=carbs_g,
                demo_user=user_id,
            )

            st.success("Meal saved to SQLite!")
            st.rerun()


# Show meals saved in SQLite
st.header("🗃️ My saved meals")

saved_meals = get_meals(demo_user=user_id)

if saved_meals:
    saved_df = pd.DataFrame(saved_meals)
    st.dataframe(
        saved_df[
            ["id", "timestamp", "food", "description", "portion_g", "carbs_g"]
        ],
        hide_index=True,
        width="stretch",
    )

    meal_to_delete = st.selectbox(
        "Select a meal to delete",
        saved_meals,

        format_func=lambda meal: (
            f"{meal['food']} — {meal['timestamp']}"
        ),
    )

    if st.button("Delete selected meal"):
        delete_meal(meal_to_delete["id"], demo_user=user_id)
        st.success("Meal deleted.")
        st.rerun()
else:
    st.info("No saved meals yet. Add one using the form above.")

# ---- P0: upload CGM CSV and log meals (stored in data/user_uploads.db) ----
st.header("Upload CGM data and log meals")
st.caption(
    "Uploaded data is stored separately from the demo CSVs. "
    "All values are still treated as simulated demo data. Not medical advice."
)

# Show the result of the last upload / meal save (kept across st.rerun)
_last_upload = st.session_state.pop("last_upload", None)
if _last_upload:
    saved = _last_upload["saved"]
    st.success(
        f"CGM file saved: {saved['readings_new']} new, "
        f"{saved['readings_updated']} updated, "
        f"{saved['readings_unchanged']} unchanged, "
        f"{saved['rows_excluded']} excluded."
    )
    for line in _last_upload["lines"]:
        st.write(line)
    if not _last_upload["excluded"].empty:
        st.dataframe(_last_upload["excluded"], hide_index=True, width="stretch")

_last_meal = st.session_state.pop("last_meal_save", None)
if _last_meal:
    st.success(
        f"Meal {_last_meal['action']}: {', '.join(_last_meal['foods'])} "
        f"at {_last_meal['meal_time']}"
    )

st.subheader("Upload a CGM CSV")
st.caption(
    "Required columns: timestamp (ISO 8601, e.g. 2026-10-20T07:45:00-07:00) "
    "and glucose_mg_dl. Times without a UTC offset are read as "
    f"{DEFAULT_TZ}. Unusable rows are excluded and reported."
)
cgm_file = st.file_uploader("CGM CSV file", type=["csv"], key="cgm_csv_upload")
if cgm_file is not None and st.button("Import CGM file", key="import_cgm_btn"):
    try:
        import_result = parse_cgm_csv(cgm_file)
        if not import_result.is_usable:
            st.error("No usable rows in this file.")
            for line in import_result.report_lines():
                st.write(line)
        else:
            saved = upload_store.save_cgm_upload(
                user_id, import_result, filename=cgm_file.name
            )
            st.session_state["last_upload"] = {
                "saved": saved,
                "lines": import_result.report_lines(),
                "excluded": import_result.excluded_frame(),
            }
            st.rerun()
    except CgmImportError as e:
        st.error(str(e))
    except ValueError as e:
        st.error(f"Could not save the file: {e}")

st.subheader("Log a meal for analysis")
with st.form("log_meal_form"):
    um_date = st.date_input("Meal date", key="um_date")
    um_time = st.time_input("Meal time", key="um_time")
    um_foods = st.text_input(
        "Foods (comma separated)",
        placeholder="e.g. white rice, grilled chicken, kimchi",
        key="um_foods",
    )
    um_submit = st.form_submit_button("Save meal for analysis")
    if um_submit:
        foods_list = [f.strip() for f in um_foods.split(",") if f.strip()]
        if not foods_list:
            st.error("Please enter at least one food.")
        else:
            try:
                st.session_state["last_meal_save"] = upload_store.save_meal(
                    user_id,
                    datetime.combine(
                        um_date, um_time.replace(second=0, microsecond=0)
                    ),
                    foods_list,
                )
                st.rerun()
            except ValueError as e:
                st.error(f"Could not save the meal: {e}")

# Uploaded meals with their cached per-meal analysis
uploaded_meals = upload_store.list_meals(user_id)
uploaded_by_id = {
    f"{user_id}_UP{m['meal_id']:05d}": m for m in uploaded_meals
}

if uploaded_meals:
    st.subheader("Per-meal results (uploaded meals)")
    st.caption(
        "Each meal has one response for the whole meal. It is not split among "
        "foods and does not show that a food caused the change."
    )
    result_rows = []
    for m in uploaded_meals:
        result_rows.append(
            {
                "meal_time": m["meal_time"],
                "foods": ", ".join(m["foods"]),
                "status": m["status"],
                "baseline_mg_dl": m["baseline_mg_dl"],
                "peak_mg_dl": m["peak_mg_dl"],
                "rise_mg_dl": m["rise_mg_dl"],
                "readings_before": m["pre_meal_readings"],
                "readings_after": m["post_meal_readings"],
                "overlap_caution": "yes" if m["warnings"] else "",
            }
        )
    st.dataframe(pd.DataFrame(result_rows), hide_index=True, width="stretch")

    to_delete = st.selectbox(
        "Select an uploaded meal to delete",
        uploaded_meals,
        format_func=lambda m: f"{', '.join(m['foods'])} - {m['meal_time']}",
        key="delete_uploaded_meal_select",
    )
    if st.button("Delete selected uploaded meal", key="delete_uploaded_meal_btn"):
        upload_store.delete_meal(user_id, to_delete["meal_id"])
        st.rerun()

with st.expander("Upload history"):
    _hist = upload_store.get_upload_history(user_id, limit=10)
    if _hist:
        st.dataframe(
            pd.DataFrame(_hist)[
                ["upload_id", "filename", "uploaded_at", "rows_read",
                 "rows_imported", "rows_excluded", "readings_new",
                 "readings_updated", "readings_unchanged"]
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No uploads yet.")

# 2. Load meal history from the analysis module
st.header("🍽️ Meal history")

try:
    cgm_all, meals_all = upload_store.combined_frames(user_id)
    history = get_meal_history(user_id, cgm=cgm_all, meals=meals_all)

    if not history:
        st.warning("No meal records found for this user.")
        st.stop()

except Exception as e:
    st.error(f"Could not load meal history: {e}")
    st.stop()

def _local_date(ts):
    return pd.to_datetime(ts, utc=True).tz_convert(DEFAULT_TZ).date()


all_dates = sorted({_local_date(h["timestamp"]) for h in history})
picked_dates = st.multiselect(
    "Dates to show",
    all_dates,
    default=all_dates,
    format_func=lambda d: d.isoformat(),
)
if picked_dates:
    history = [h for h in history if _local_date(h["timestamp"]) in picked_dates]
else:
    st.info("No date selected; showing all dates.")

# Old comparison functions have no "enough data" gate, so leave out uploaded
# meals that meal_analysis marked insufficient_data.
_insufficient_ids = {
    mid for mid, m in uploaded_by_id.items() if m["status"] != "ok"
}
meals_cmp = meals_all[~meals_all["meal_id"].isin(_insufficient_ids)]

def meal_label(meal):
    food = str(meal.get("food", "Unknown food"))
    timestamp = str(meal.get("timestamp", "Unknown time"))
    return f"{food.replace('_', ' ').title()} — {timestamp}"


selected_meal = st.selectbox(
    "Choose a meal",
    history,
    format_func=lambda m: meal_label(m)
    + (" [uploaded]" if m["meal_id"] in uploaded_by_id else " [generated]"),
)

meal_id = selected_meal["meal_id"]

st.write("**Food:**", str(selected_meal.get("food", "Unknown")).replace("_", " ").title())
st.write("**Meal time:**", selected_meal.get("timestamp", "Unknown"))
st.write("**Description:**", selected_meal.get("description", "No description available"))
_portion = selected_meal.get("primary_portion_g")
st.write(
    "**Portion:**",
    "N/A" if _portion is None or pd.isna(_portion) else f"{_portion} g",
)

# 3. Get glucose analysis
st.header("🔎 Glucose analysis")

try:
    insufficient_note = None
    _info = uploaded_by_id.get(meal_id)
    if _info is not None and _info["status"] != "ok":
        result = None
        insufficient_note = _info["message"]
    else:
        result = get_meal_glucose(user_id, meal_id, cgm=cgm_all, meals=meals_all)

    if result is None:
        st.warning(
            insufficient_note
            or "No CGM data is available for this meal's analysis window."
        )
    else:
        baseline = result.get("baseline")
        peak = result.get("peak")
        rise = result.get("rise")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Baseline",
            f"{baseline:.1f} mg/dL" if baseline is not None else "N/A",
        )
        col2.metric(
            "Peak",
            f"{peak:.1f} mg/dL" if peak is not None else "N/A",
        )
        col3.metric(
            "Glucose rise",
            f"{rise:+.1f} mg/dL" if rise is not None else "N/A",
        )

        st.caption(
            "Baseline: average glucose during the 15 minutes before the meal. "
            "Peak: maximum glucose within 2 hours after the meal. "
            "Rise: peak minus baseline. Values are simulated."
        )

except Exception as e:
    st.error(f"Could not calculate meal glucose metrics: {e}")
    result = None

_info = uploaded_by_id.get(meal_id)
if _info is not None:
    for w in _info["warnings"]:
        st.warning(w["message"])
    st.caption(_info["interpretation_note"])

# 4. Glucose curve
st.subheader("📈 Glucose after the meal")

try:
    curve = get_glucose_curve(user_id, meal_id, cgm=cgm_all, meals=meals_all)

    if curve:
        curve_df = pd.DataFrame(curve)

        if not curve_df.empty and {
            "minutes_from_meal",
            "glucose_mg_dl",
        }.issubset(curve_df.columns):

            fig = px.line(
                curve_df,
                x="minutes_from_meal",
                y="glucose_mg_dl",
                markers=True,
                labels={
                    "minutes_from_meal": "Minutes from meal",
                    "glucose_mg_dl": "Glucose (mg/dL)",
                },
                title="Glucose curve around the meal",
            )

            fig.add_vline(
                x=0,
                line_dash="dash",
                annotation_text="Meal",
            )

            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, width="stretch")

        else:
            st.warning("The glucose curve has an unexpected data format.")
    else:
        st.warning("No glucose curve is available for this meal.")

except Exception as e:
    st.error(f"Could not load glucose curve: {e}")

# 5. Compare foods
with st.expander("🥗 Compare foods"):
    try:
        comparisons = compare_foods(user_id, cgm=cgm_all, meals=meals_cmp)

        if comparisons:
            comparison_df = pd.DataFrame(comparisons)
            st.dataframe(comparison_df, width="stretch")
        else:
            st.info("No food comparison data available.")

    except Exception as e:
        st.error(f"Could not compare foods: {e}")

with st.expander("Foods in uploaded meals (whole-meal averages)"):
    if uploaded_meals:
        st.caption(
            "Each food is listed with the average of the meals that contained "
            "it. This is not the effect of the food itself."
        )
        st.dataframe(
            pd.DataFrame(summarize_by_food(uploaded_meals)),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No uploaded meals yet.")

# 6. User patterns
with st.expander("📊 User glucose patterns"):
    try:
        patterns = analyze_user_patterns(user_id, cgm=cgm_all, meals=meals_cmp)
                st.json(json.loads(json.dumps(patterns), parse_constant=lambda c: None))

    except Exception as e:
        st.error(f"Could not analyze user patterns: {e}")

# 7. Ask Foundry Agent (chat)
st.header("🤖 Ask GlucoWise AI")

# Reset the chat when a different user logs in
if st.session_state.get("chat_user") != user_id:
    st.session_state["chat_user"] = user_id
    st.session_state["chat_messages"] = []

messages = st.session_state["chat_messages"]

for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("e.g. How did white rice compare with brown rice?")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    # Recent turns are added as context only; numbers must come from the JSON.
    recent = messages[-6:]
    full_question = question
    if recent:
        history_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent
        )
        full_question = (
            f"{question}\n\nEarlier conversation (context only; "
            f"take every number from the JSON above, never from this text):\n"
            f"{history_text}"
        )

    with st.chat_message("assistant"):
        try:
            prompt = build_agent_prompt(user_id, full_question)

            with st.expander("🔍 Prompt sent to the AI (debug)"):
                st.code(prompt, language="json")

            if USE_MOCK:
                answer = (
                    f"(mock answer) turn {len(messages) // 2 + 1}, "
                    f"prompt length {len(prompt)} chars"
                )
            else:
                # Endpoint lives in .streamlit/secrets.toml (git-ignored).
                endpoint = st.secrets["FOUNDRY_ENDPOINT"]
                agent_name = "GlucoWise-Agent"

                with st.spinner("GlucoWise AI is thinking..."):
                    with DefaultAzureCredential() as credential:
                        with AIProjectClient(
                            endpoint=endpoint,
                            credential=credential,
                            allow_preview=True,
                        ) as project:
                            openai_client = project.get_openai_client(
                                agent_name=agent_name
                            )
                            response = openai_client.responses.create(
                                input=prompt
                            )
                answer = response.output_text

            st.markdown(answer)
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Foundry Agent request failed: {e}")

if messages and st.button("Clear chat"):
    st.session_state["chat_messages"] = []
    st.rerun()