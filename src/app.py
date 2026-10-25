
import base64
import io
import json
import os
import re
from datetime import datetime, time as dtime
from auth_db import login_gate

import pandas as pd
import plotly.express as px
import streamlit as st
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from food_recognition import recognize_foods, normalize_food_text
from normalization import normalize_food

from agent_context import build_agent_prompt
from glucose_analysis import (
    compare_foods,
    get_glucose_curve,
    get_meal_glucose,
    get_meal_history,
    load_data,
)
from local_db import delete_meal, get_meals, save_meal
from user_data import (
    TIMEZONE,
    cgm_summary,
    check_meals,
    init_user_tables,
    parse_cgm_csv,
    reset_user,
    save_cgm,
    with_user_data,
)

ENDPOINT = os.getenv(
    "FOUNDRY_ENDPOINT",
    "https://whb050704-4696-resource.services.ai.azure.com/api/projects/whb050704-4696",
)
AGENT_NAME = os.getenv("FOUNDRY_AGENT", "GlucoWise-Agent")
AUTO_QUESTION = (
    "Summarize what the data shows about my glucose rise for each food, meal by meal. "
    "State how many meals each food has and say clearly when the sample is too small. "
    "Do not claim that a food caused a rise."
)

st.set_page_config(page_title="GlucoWise", page_icon="📈", layout="wide")

# ----------------------------------------------------------------- helpers
@st.cache_data
def load_base():
    return load_data()


def flash(message: str) -> None:
    """Notification that remains visible after st.rerun()."""
    st.session_state["flash"] = message


def ask_agent(agent_input):
    with DefaultAzureCredential() as credential:
        with AIProjectClient(
            endpoint=ENDPOINT, credential=credential, allow_preview=True
        ) as project:
            client = project.get_openai_client(agent_name=AGENT_NAME)
            return client.responses.create(input=agent_input).output_text


def meal_label(meal):
    food = str(meal.get("food", "Unknown food")).replace("_", " ").title()
    return f"{food} — {str(meal.get('timestamp', ''))[:16].replace('T', ' ')}"


# -------------------------------------------------------------------- page
init_user_tables()

st.title("📈 GlucoWise")
st.caption("Understand how meals affect your glucose levels")
st.info(
    "Demo mode: all glucose data shown here are simulated. "
    "This app does not provide medical advice and does not claim that a food causes a glucose response."
)

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

user_id = login_gate()

base_cgm, base_meals = load_base()
cgm, meals = with_user_data(base_cgm, base_meals, user_id)
history = get_meal_history(user_id, cgm=cgm, meals=meals)

selected_meal, meal_id, result = None, None, None
upload_round = st.session_state.setdefault("upload_round", 0)

tab_in, tab_day, tab_pattern, tab_ai = st.tabs(
    ["1. Data Input", "2. Meal Analysis", "3. Historical Patterns", "4. AI Insights"]
)

# ============================================================ 1. input
with tab_in:
    st.subheader("① Daily CGM Data")
    summary = cgm_summary(user_id)
    if summary["rows"]:
        st.caption(
            f"Uploaded CGM data: {summary['days']} days · {summary['rows']} readings "
            f"({summary['first'][:10]} ~ {summary['last'][:10]})"
        )

    cgm_file = st.file_uploader("Upload CGM CSV", type="csv", key=f"cgm_file_{upload_round}")
    if cgm_file is not None:
        try:
            parsed = parse_cgm_csv(io.BytesIO(cgm_file.getvalue()))
            local = parsed["time_utc"].dt.tz_convert(TIMEZONE)
            st.write(
                f"{len(parsed)} readings · {local.min():%Y-%m-%d %H:%M} ~ {local.max():%Y-%m-%d %H:%M}"
            )
            if st.button("Save CGM Data", key="save_cgm"):
                saved_n = save_cgm(user_id, parsed)
                st.session_state["upload_round"] += 1
                flash(f"Saved {saved_n} CGM readings.")
                st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.subheader("② Food Photos and Meal Times")
    default_day = (
        pd.Timestamp(summary["last"]).date() if summary["last"] else datetime.now().date()
    )
    photos = st.file_uploader(
        "Upload Food Photos (multiple files allowed)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"photos_{upload_round}",
    )
    if photos:
        keys = [f"{p.name}-{p.size}-{upload_round}" for p in photos]

        if st.button("🤖 Recognize Food with AI", key="recognize"):
            with st.spinner("Recognizing food in your photos..."):
                for p, k in zip(photos, keys):
                    try:
                        items = recognize_foods(p.getvalue(), p.type or "image/jpeg")
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"{p.name}: Could not recognize food ({exc}). Please enter it manually.")
                        continue
                    if items:
                        st.session_state[f"desc_{k}"] = ", ".join(items)
                        st.session_state[f"ai_text_{k}"] = ", ".join(items)
                        st.session_state[f"ai_label_{k}"] = "_".join(items)

        entries = []
        for p, k in zip(photos, keys):
            col_img, col_form = st.columns([1, 3])
            col_img.image(p.getvalue(), width=140)
            with col_form:
                day = st.date_input("Meal Date", value=default_day, key=f"date_{k}")
                clock = st.time_input("Meal Time", value=dtime(12, 0), key=f"time_{k}")
                desc = st.text_input("All Detected Foods / Notes", key=f"desc_{k}")
            entries.append({"key": k, "day": day, "clock": clock, "desc": desc})
            st.divider()

        if st.button("Save Meal Records", type="primary", key="save_meals"):
            if any(not e["desc"].strip() for e in entries):
                st.error("Please enter food information for all meals.")
            else:
                labels = []
                try:
                    with st.spinner("Normalizing food names..."):
                        for e in entries:
                            text = e["desc"].strip()
                            ai_text = st.session_state.get(f"ai_text_{e['key']}")
                            if ai_text is not None and text == ai_text:
                                label = normalize_food(st.session_state[f"ai_label_{e['key']}"])
                            else:
                                label = normalize_food_text(text)
                            labels.append(label)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Food name normalization failed ({exc}). Nothing was saved; please try again.")
                else:
                    for e, label in zip(entries, labels):
                        save_meal(
                            timestamp=datetime.combine(e["day"], e["clock"]).isoformat(timespec="minutes"),
                            food=label,
                            description=e["desc"].strip(),
                            portion_g=None,
                            carbs_g=None,
                            demo_user=user_id,
                        )
                    st.session_state["upload_round"] += 1
                    flash(f"Saved {len(entries)} meal record(s).")
                    st.rerun()

    st.subheader("③ Saved Meals")
    saved = get_meals(demo_user=user_id)
    if saved:
        st.dataframe(
            pd.DataFrame(saved)[["id", "timestamp", "food", "description"]],
            hide_index=True,
            width="stretch",
        )
        by_id = {m["id"]: m for m in saved}
        del_id = st.selectbox(
            "Select a Meal to Delete",
            list(by_id),
            format_func=lambda i: f"{by_id[i]['food']} — {by_id[i]['timestamp']}",
        )
        if st.button("Delete Selected Meal"):
            delete_meal(del_id, demo_user=user_id)
            flash("Meal deleted.")
            st.rerun()
    else:
        st.info("No saved meals yet. Upload a food photo and enter the meal time above.")

    with st.expander("⚠️ Reset Demo Data"):
        st.caption(f"Delete all uploaded CGM data and meal records for account '{user_id}'. (Sample CSV data will remain.)")
        if st.checkbox("I understand; delete my data", key="confirm_reset") and st.button("Reset Data"):
            reset_user(user_id)
            flash("Data reset successfully.")
            st.rerun()

# ======================================================= 2. per-meal view
with tab_day:
    user_cgm = cgm[cgm["user_id"] == user_id]
    local_times = user_cgm["time"].dt.tz_convert(TIMEZONE)
    cgm_days = set(local_times.dt.date)
    meal_days = {pd.Timestamp(str(h["timestamp"])[:10]).date() for h in history}
    days = sorted(cgm_days | meal_days, reverse=True)
    if not days:
        st.info("No CGM data or meal records found. Upload your data in the 'Data Input' tab.")
    else:
        day_sel = st.selectbox("Date", days, format_func=str)
        day_cgm = user_cgm[local_times.dt.date == day_sel]
        day_meals = [h for h in history if str(h["timestamp"])[:10] == str(day_sel)]

        if day_cgm.empty:
            st.warning(
                f"No CGM data available for {day_sel}. Meals are recorded, but glucose cannot be calculated. "
                "Upload the CGM file for this date in the 'Data Input' tab."
            )
        else:
            plot_df = pd.DataFrame(
                {
                    "time": day_cgm["time"].dt.tz_convert(TIMEZONE).dt.tz_localize(None),
                    "glucose": day_cgm["glucose_mg_dl"],
                }
            )
            fig = px.line(
                plot_df,
                x="time",
                y="glucose",
                labels={"time": "Time", "glucose": "Glucose (mg/dL)"},
                title=f"{day_sel} CGM (simulated)",
            )
            for h in day_meals:
                x = pd.Timestamp(h["timestamp"]).tz_convert(TIMEZONE).tz_localize(None)
                fig.add_shape(
                    type="line", x0=x, x1=x, y0=0, y1=1, yref="paper",
                    line=dict(dash="dash", width=1),
                )
                fig.add_annotation(
                    x=x, y=1, yref="paper", yanchor="bottom", showarrow=False,
                    text=str(h["food"]).replace("_", " "),
                )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, width="stretch")

        if not day_meals:
            st.info("No meal records for this date.")
        else:
            warns = check_meals(cgm, meals, user_id)
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Time": str(h["timestamp"])[11:16],
                            "Food": str(h["food"]).replace("_", " "),
                            "Baseline": h["baseline"],
                            "Peak": h["peak"],
                            "Rise": h["rise"],
                            "Notes": " / ".join(warns.get(h["meal_id"], [])),
                        }
                        for h in day_meals
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

            selected_meal = st.selectbox("Select a Meal to View Details", day_meals, format_func=meal_label)
            meal_id = selected_meal["meal_id"]
            result = get_meal_glucose(user_id, meal_id, cgm=cgm, meals=meals)

            if result is None:
                st.warning("There is not enough CGM data in the analysis window for this meal.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Baseline", f"{result['baseline']:.1f} mg/dL")
                c2.metric("Peak", f"{result['peak']:.1f} mg/dL")
                c3.metric("Glucose Rise", f"{result['rise']:+.1f} mg/dL")
                st.caption(
                    "Baseline: average glucose during the 15 minutes before the meal. "
                    "Peak: maximum glucose within 2 hours after the meal. "
                    "Rise: peak minus baseline. Values are simulated."
                )

            curve = get_glucose_curve(user_id, meal_id, cgm=cgm, meals=meals)
            if curve:
                fig2 = px.line(
                    pd.DataFrame(curve),
                    x="minutes_from_meal",
                    y="glucose_mg_dl",
                    markers=True,
                    labels={"minutes_from_meal": "Minutes from meal", "glucose_mg_dl": "Glucose (mg/dL)"},
                    title="Glucose Curve Around the Meal",
                )
                fig2.add_vline(x=0, line_dash="dash", annotation_text="Meal")
                fig2.update_layout(hovermode="x unified")
                st.plotly_chart(fig2, width="stretch")

# ===================================================== 3. accumulated
with tab_pattern:
    comparisons = compare_foods(user_id, cgm=cgm, meals=meals)
    skipped = [h for h in history if h["rise"] is None]
    if skipped:
        st.warning(
            f"{len(skipped)} meal record(s) excluded from analysis due to missing CGM data: "
            + ", ".join(f"{str(h['food']).replace('_', ' ')} ({str(h['timestamp'])[:16].replace('T', ' ')})" for h in skipped)
        )
    if not comparisons:
        st.info("No meal records available for comparison yet.")
    else:
        comp_df = pd.DataFrame(comparisons)
        comp_df["Sample Size"] = comp_df["meal_count"].map(lambda n: "Small (fewer than 3 meals)" if n < 3 else "")
        st.dataframe(comp_df, hide_index=True, width="stretch")
        bar = px.bar(
            comp_df,
            x="food",
            y="average_rise_mg_dl",
            text="meal_count",
            labels={"food": "Food", "average_rise_mg_dl": "Average Rise (mg/dL)"},
            title="Average Glucose Rise by Food (text = number of meals)",
        )
        st.plotly_chart(bar, width="stretch")
        st.caption(
            "Meal size, side dishes, exercise, sleep, and other factors can affect glucose levels. "
            "This table shows observed patterns in your records and does not mean that a specific food caused the response."
        )

# ============================================================== 4. AI
with tab_ai:
    st.caption(
        "AI explains results calculated by Python. "
        "Glucose values are based on the currently selected user's records."
    )

    # Reset conversation when the user account changes
    if st.session_state.get("chat_user_id") != user_id:
        st.session_state["chat_user_id"] = user_id
        st.session_state["chat_messages"] = []

    messages = st.session_state["chat_messages"]

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("💬 GlucoWise AI")
    with col2:
        if st.button("Clear Conversation", key="clear_ai_chat"):
            st.session_state["chat_messages"] = []
            st.rerun()

    # Display previous messages
    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Keep the automatic insights button
    auto_clicked = st.button(
        "📊 Summarize Automatic Insights",
        key="auto_chat_insight",
    )

    question = st.chat_input(
        "e.g., Explain glucose changes on days I ate white rice",
        key="ai_chat_input",
    )

    if question or auto_clicked:
        q = question.strip() if question else AUTO_QUESTION

        if q:
            # Get some previous messages before adding the current question
            previous_messages = st.session_state["chat_messages"][-8:]

            history_text = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: "
                f"{m['content']}"
                for m in previous_messages
            )

            prefix = f"(Selected meal: {meal_id}) " if meal_id else ""

            # Use existing Python glucose analysis results in the prompt
            data_prompt = build_agent_prompt(
                user_id,
                prefix + q,
                cgm=cgm,
                meals=meals,
            ).replace("NaN", "null")

            # Send previous conversation and current question to the AI
            prompt = f"""
You are continuing a GlucoWise conversation.

Previous conversation:
{history_text if history_text else "(No previous conversation)"}

Use the previous conversation for context.
Answer the current question using the GlucoWise data
and analysis instructions below. Do not invent glucose values.

Current GlucoWise request and data:
{data_prompt}
"""

            # Display and save the user's message
            st.session_state["chat_messages"].append(
                {"role": "user", "content": q}
            )

            with st.chat_message("user"):
                st.markdown(q)

            try:
                with st.chat_message("assistant"):
                    with st.spinner("GlucoWise AI is thinking..."):
                        answer = ask_agent(prompt)

                    st.markdown(answer)

                st.session_state["chat_messages"].append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as exc:
                st.error(f"Foundry Agent request failed: {exc}")
