
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="GlucoWise",
    page_icon="📈",
    layout="wide",
)

st.title("📈 GlucoWise")
st.caption("Understand how meals affect your glucose levels")

st.info(
    "Demo mode: uploaded data is used for analysis. "
    "This app does not provide medical advice."
)

# 1. Meal input
st.header("🍽️ Log your meal")

col1, col2 = st.columns(2)

with col1:
    meal_name = st.text_input(
        "Food name",
        placeholder="e.g. Chicken rice bowl",
    )

with col2:
    meal_time = st.time_input("Meal time")

# 2. CGM CSV upload
st.header("📂 Upload CGM data")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="CSV should contain a timestamp column and a glucose column.",
)

df = None
time_col = None
glucose_col = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Accept common column-name variations
        columns = {str(c).strip().lower(): c for c in df.columns}

        time_candidates = [
            "timestamp", "time", "datetime", "date_time", "date"
        ]
        glucose_candidates = [
            "glucose", "glucose_mg_dl", "glucose_mg/dl",
            "blood_glucose", "value", "mg/dl"
        ]

        time_col = next(
            (columns[c] for c in time_candidates if c in columns),
            None,
        )
        glucose_col = next(
            (columns[c] for c in glucose_candidates if c in columns),
            None,
        )

        if time_col is None or glucose_col is None:
            st.error(
                "Could not identify the required columns. "
                "Please use columns named timestamp and glucose."
            )
            st.write("Columns found:", list(df.columns))
            df = None
        else:
            df[time_col] = pd.to_datetime(
                df[time_col], errors="coerce"
            )
            df[glucose_col] = pd.to_numeric(
                df[glucose_col], errors="coerce"
            )
            df = df.dropna(subset=[time_col, glucose_col])

            if df.empty:
                st.error("No valid timestamp and glucose readings found.")
            else:
                st.success(f"CSV loaded: {len(df)} valid readings")

                with st.expander("Preview uploaded data"):
                    st.dataframe(df.head(10), use_container_width=True)

    except Exception as e:
        st.error(f"Could not read the CSV: {e}")
        df = None

# 3. Analyze button
st.header("🔎 Meal analysis")

if st.button("Analyze my meal", type="primary"):
    if not meal_name.strip():
        st.warning("Please enter a food name.")
    elif uploaded_file is None:
        st.warning("Please upload a CGM CSV file.")
    elif df is None or df.empty:
        st.warning("Please upload a valid CSV file.")
    else:
        st.session_state["meal_name"] = meal_name.strip()
        st.session_state["meal_time"] = str(meal_time)
        st.session_state["cgm_data"] = df
        st.success("Meal and CGM data saved for this session!")

# 4. Glucose chart and output
if "cgm_data" in st.session_state:
    st.subheader("📈 Glucose over time")

    chart_df = st.session_state["cgm_data"].sort_values(time_col)

    fig = px.line(
        chart_df,
        x=time_col,
        y=glucose_col,
        markers=True,
        labels={
            time_col: "Time",
            glucose_col: "Glucose (mg/dL)",
        },
        title="Uploaded CGM readings",
    )
    fig.update_layout(hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Meal details")
    st.write("**Food:**", st.session_state["meal_name"])
    st.write("**Meal time:**", st.session_state["meal_time"])

    st.subheader("🤖 Ask AI about your meal")

    question = st.text_input(
        "Your question",
        placeholder="e.g. What happened to my glucose after this meal?",
    )

    if st.button("Ask AI"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            st.info(
                "AI integration is not connected yet. "
                "This section will display the answer after "
                "the Microsoft Foundry integration is added."
            )
