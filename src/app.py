
import streamlit as st

st.set_page_config(
    page_title="GlucoWise",
    page_icon="📈",
    layout="wide",
)

st.title("📈 GlucoWise")
st.subheader("Understand how meals affect your glucose")

st.info(
    "Demo mode: This app uses simulated glucose data, "
    "not real medical measurements."
)

st.header("Log your meal")

meal_name = st.text_input(
    "What did you eat?",
    placeholder="e.g. Chicken rice bowl",
)

meal_time = st.time_input("When did you eat?")

if st.button("Analyze my meal"):
    if not meal_name.strip():
        st.warning("Please enter a meal name.")
    else:
        st.success(f"Ready to analyze: {meal_name} at {meal_time}")
        st.write("Next: connect the meal to glucose analysis.")
