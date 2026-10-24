import streamlit as st
import pandas as pd
import plotly.express as px
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

from glucose_analysis import (
    get_meal_history,
    get_meal_glucose,
    get_glucose_curve,
    compare_foods,
    analyze_user_patterns,
)
from agent_context import build_agent_prompt

st.set_page_config(
    page_title="GlucoWise",
    page_icon="📈",
    layout="wide",
)

st.title("📈 GlucoWise")
st.caption("Understand how meals affect your glucose levels")

st.info(
    "Demo mode: All glucose data shown here are simulated. "
    "This app does not provide medical advice."
)

# 1. Select user
st.header("👤 Select user")

user_id = st.selectbox(
    "User",
    ["A", "B", "C"],
)

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
            st.plotly_chart(fig, use_container_width=True)

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
            st.dataframe(comparison_df, use_container_width=True)
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

# 7. Ask Foundry Agent
st.header("🤖 Ask GlucoWise AI")

question = st.text_input(
    "Your question",
    placeholder="e.g. How did white rice compare with brown rice?",
)

if st.button("Ask AI", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            # Full pre-computed analysis for the selected user + the question.
            # The agent only explains these numbers.
            prompt = build_agent_prompt(user_id, question)

            # Debug view: shows exactly what the agent receives (works without Azure).
            print(prompt)
            with st.expander("🔍 Prompt sent to the AI (debug)"):
                st.code(prompt, language="json")

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

            st.session_state["ai_answer"] = response.output_text

        except Exception as e:
            st.error(f"Foundry Agent request failed: {e}")

if "ai_answer" in st.session_state:
    st.subheader("💬 GlucoWise AI response")
    st.markdown(st.session_state["ai_answer"])