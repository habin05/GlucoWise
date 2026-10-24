import os

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


# 2. Load meal history from the analysis module
st.header("🍽️ Meal history")

try:
    history = get_meal_history(user_id)

    if not history:
        st.warning("No meal records found for this user.")
        st.stop()

except Exception as e:
    st.error(f"Could not load meal history: {e}")
    st.stop()


def meal_label(meal):
    food = str(meal.get("food", "Unknown food"))
    timestamp = str(meal.get("timestamp", "Unknown time"))
    return f"{food.replace('_', ' ').title()} — {timestamp}"


selected_meal = st.selectbox(
    "Choose a meal",
    history,
    format_func=meal_label,
)

meal_id = selected_meal["meal_id"]

st.write("**Food:**", str(selected_meal.get("food", "Unknown")).replace("_", " ").title())
st.write("**Meal time:**", selected_meal.get("timestamp", "Unknown"))
st.write("**Description:**", selected_meal.get("description", "No description available"))
st.write("**Portion:**", f"{selected_meal.get('primary_portion_g', 'N/A')} g")

# 3. Get glucose analysis
st.header("🔎 Glucose analysis")

try:
    result = get_meal_glucose(user_id, meal_id)

    if result is None:
        st.warning(
            "No CGM data is available for this meal's analysis window."
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

# 4. Glucose curve
st.subheader("📈 Glucose after the meal")

try:
    curve = get_glucose_curve(user_id, meal_id)

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
        comparisons = compare_foods(user_id)

        if comparisons:
            comparison_df = pd.DataFrame(comparisons)
            st.dataframe(comparison_df, width="stretch")
        else:
            st.info("No food comparison data available.")

    except Exception as e:
        st.error(f"Could not compare foods: {e}")

# 6. User patterns
with st.expander("📊 User glucose patterns"):
    try:
        patterns = analyze_user_patterns(user_id)
        st.json(patterns)

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