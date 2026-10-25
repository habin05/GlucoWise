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
    """st.rerun() 뒤에도 보이는 알림."""
    st.session_state["flash"] = message


def ask_agent(agent_input):
    with DefaultAzureCredential() as credential:
        with AIProjectClient(
            endpoint=ENDPOINT, credential=credential, allow_preview=True
        ) as project:
            client = project.get_openai_client(agent_name=AGENT_NAME)
            return client.responses.create(input=agent_input).output_text


def recognize_foods(image_bytes: bytes, mime: str) -> list[str]:
    """사진 속 음식 이름 목록. 확인 안 되는 재료는 추측하지 않도록 요청한다.

    NOTE: Foundry 에이전트에 이미지를 data URL로 넘기는 방식(Responses API)이다.
    팀에서 이미 성공한 호출 방식이 있으면 이 함수 안만 그 코드로 바꾸면 된다.
    """
    b64 = base64.b64encode(image_bytes).decode()
    text = ask_agent(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "List the foods visible in this photo. Reply ONLY with a JSON "
                            'array of short food names, e.g. ["white rice", "egg"]. '
                            "Do not guess ingredients you cannot see."
                        ),
                    },
                    {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
                ],
            }
        ]
    )
    match = re.search(r"\[.*?\]", text, re.S)
    if not match:
        return []
    try:
        return [str(x).strip() for x in json.loads(match.group(0)) if str(x).strip()]
    except json.JSONDecodeError:
        return []


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
    ["1. 데이터 입력", "2. 식사별 분석", "3. 누적 패턴", "4. AI 설명"]
)

# ============================================================ 1. input
with tab_in:
    st.subheader("① 하루치 CGM 파일")
    summary = cgm_summary(user_id)
    if summary["rows"]:
        st.caption(
            f"저장된 업로드 CGM: {summary['days']}일 · {summary['rows']}개 값 "
            f"({summary['first'][:10]} ~ {summary['last'][:10]})"
        )

    cgm_file = st.file_uploader("CGM CSV 업로드", type="csv", key=f"cgm_file_{upload_round}")
    if cgm_file is not None:
        try:
            parsed = parse_cgm_csv(io.BytesIO(cgm_file.getvalue()))
            local = parsed["time_utc"].dt.tz_convert(TIMEZONE)
            st.write(
                f"{len(parsed)}개 값 · {local.min():%Y-%m-%d %H:%M} ~ {local.max():%Y-%m-%d %H:%M}"
            )
            if st.button("CGM 저장", key="save_cgm"):
                saved_n = save_cgm(user_id, parsed)
                st.session_state["upload_round"] += 1
                flash(f"CGM {saved_n}개 값을 저장했어요.")
                st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.subheader("② 음식 사진과 식사 시간")
    default_day = (
        pd.Timestamp(summary["last"]).date() if summary["last"] else datetime.now().date()
    )
    photos = st.file_uploader(
        "음식 사진 (여러 장 가능)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"photos_{upload_round}",
    )
    if photos:
        keys = [f"{p.name}-{p.size}-{upload_round}" for p in photos]

        if st.button("🤖 AI로 음식 인식", key="recognize"):
            with st.spinner("사진 속 음식을 인식하는 중..."):
                for p, k in zip(photos, keys):
                    try:
                        items = recognize_foods(p.getvalue(), p.type or "image/jpeg")
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"{p.name}: 인식하지 못했어요 ({exc}). 직접 입력해 주세요.")
                        continue
                    if items:
                        st.session_state[f"main_{k}"] = items[0]
                        st.session_state[f"desc_{k}"] = ", ".join(items)

        entries = []
        for p, k in zip(photos, keys):
            col_img, col_form = st.columns([1, 3])
            col_img.image(p.getvalue(), width=140)
            with col_form:
                day = st.date_input("식사 날짜", value=default_day, key=f"date_{k}")
                clock = st.time_input("식사 시간", value=dtime(12, 0), key=f"time_{k}")
                main = st.text_input("주요 음식 (누적 비교 기준)", key=f"main_{k}")
                desc = st.text_input("인식된 음식 전체 / 메모", key=f"desc_{k}")
            entries.append({"day": day, "clock": clock, "main": main, "desc": desc})
            st.divider()

        if st.button("식사 기록 저장", type="primary", key="save_meals"):
            if any(not e["main"].strip() for e in entries):
                st.error("주요 음식 이름을 모두 입력해 주세요.")
            else:
                for e in entries:
                    save_meal(
                        timestamp=datetime.combine(e["day"], e["clock"]).isoformat(timespec="minutes"),
                        food=e["main"].strip(),
                        description=e["desc"].strip(),
                        portion_g=None,  # 사진만으로 섭취량을 확정하지 않음
                        carbs_g=None,
                        demo_user=user_id,
                    )
                st.session_state["upload_round"] += 1
                flash(f"식사 {len(entries)}건을 저장했어요.")
                st.rerun()

    st.subheader("③ 저장된 식사")
    saved = get_meals(demo_user=user_id)
    if saved:
        st.dataframe(
            pd.DataFrame(saved)[["id", "timestamp", "food", "description"]],
            hide_index=True,
            width="stretch",
        )
        by_id = {m["id"]: m for m in saved}
        del_id = st.selectbox(
            "삭제할 식사",
            list(by_id),
            format_func=lambda i: f"{by_id[i]['food']} — {by_id[i]['timestamp']}",
        )
        if st.button("선택한 식사 삭제"):
            delete_meal(del_id, demo_user=user_id)
            flash("식사를 삭제했어요.")
            st.rerun()
    else:
        st.info("저장된 식사가 없어요. 위에서 사진과 시간을 입력해 저장하세요.")

    with st.expander("⚠️ 시연 초기화"):
        st.caption(f"'{user_id}' 계정의 업로드 CGM과 식사 기록을 모두 지웁니다. (CSV 샘플 데이터는 그대로)")
        if st.checkbox("정말 지울게요", key="confirm_reset") and st.button("초기화 실행"):
            reset_user(user_id)
            flash("초기화했어요.")
            st.rerun()

# ======================================================= 2. per-meal view
with tab_day:
    user_cgm = cgm[cgm["user_id"] == user_id]
    local_times = user_cgm["time"].dt.tz_convert(TIMEZONE)
    cgm_days = set(local_times.dt.date)
    meal_days = {pd.Timestamp(str(h["timestamp"])[:10]).date() for h in history}
    days = sorted(cgm_days | meal_days, reverse=True)
    if not days:
        st.info("CGM과 식사 기록이 없어요. '데이터 입력' 탭에서 업로드하세요.")
    else:
        day_sel = st.selectbox("날짜", days, format_func=str)
        day_cgm = user_cgm[local_times.dt.date == day_sel]
        day_meals = [h for h in history if str(h["timestamp"])[:10] == str(day_sel)]

        if day_cgm.empty:
            st.warning(
                f"{day_sel}에는 CGM 데이터가 없어요. 식사는 기록돼 있지만 혈당을 계산할 수 없어요. "
                "'데이터 입력' 탭에서 이 날짜의 CGM 파일을 업로드하세요."
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
            st.info("이 날짜에는 식사 기록이 없어요.")
        else:
            warns = check_meals(cgm, meals, user_id)
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "시간": str(h["timestamp"])[11:16],
                            "음식": str(h["food"]).replace("_", " "),
                            "Baseline": h["baseline"],
                            "Peak": h["peak"],
                            "Rise": h["rise"],
                            "유의사항": " / ".join(warns.get(h["meal_id"], [])),
                        }
                        for h in day_meals
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

            selected_meal = st.selectbox("자세히 볼 식사", day_meals, format_func=meal_label)
            meal_id = selected_meal["meal_id"]
            result = get_meal_glucose(user_id, meal_id, cgm=cgm, meals=meals)

            if result is None:
                st.warning("이 식사는 분석 구간에 CGM 데이터가 부족해 계산할 수 없어요.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Baseline", f"{result['baseline']:.1f} mg/dL")
                c2.metric("Peak", f"{result['peak']:.1f} mg/dL")
                c3.metric("Glucose rise", f"{result['rise']:+.1f} mg/dL")
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
                    title="Glucose curve around the meal",
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
            f"CGM 데이터가 없어 분석에서 제외된 식사 {len(skipped)}건: "
            + ", ".join(f"{str(h['food']).replace('_', ' ')} ({str(h['timestamp'])[:16].replace('T', ' ')})" for h in skipped)
        )
    if not comparisons:
        st.info("비교할 식사 기록이 아직 없어요.")
    else:
        comp_df = pd.DataFrame(comparisons)
        comp_df["표본"] = comp_df["meal_count"].map(lambda n: "적음 (3회 미만)" if n < 3 else "")
        st.dataframe(comp_df, hide_index=True, width="stretch")
        bar = px.bar(
            comp_df,
            x="food",
            y="average_rise_mg_dl",
            text="meal_count",
            labels={"food": "Food", "average_rise_mg_dl": "Average rise (mg/dL)"},
            title="Average glucose rise by food (text = number of meals)",
        )
        st.plotly_chart(bar, width="stretch")
        st.caption(
            "식사량, 반찬, 운동, 수면 등 여러 요인이 혈당에 영향을 줄 수 있어, "
            "이 표는 기록에서 관찰된 경향일 뿐 특정 음식이 원인이라는 뜻이 아닙니다."
        )

# ============================================================== 4. AI

# ============================================================== 4. AI
with tab_ai:
    st.caption(
        "AI는 Python이 계산한 결과를 설명합니다. "
        "혈당 수치는 현재 선택된 사용자의 기록을 기준으로 합니다."
    )

    # 사용자 계정이 바뀌면 이전 대화가 섞이지 않도록 초기화
    if st.session_state.get("chat_user_id") != user_id:
        st.session_state["chat_user_id"] = user_id
        st.session_state["chat_messages"] = []

    messages = st.session_state["chat_messages"]

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("💬 GlucoWise AI")
    with col2:
        if st.button("대화 초기화", key="clear_ai_chat"):
            st.session_state["chat_messages"] = []
            st.rerun()

    # 이전 대화 표시
    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 자동 인사이트 버튼도 유지
    auto_clicked = st.button(
        "📊 자동 인사이트 요약",
        key="auto_chat_insight",
    )

    question = st.chat_input(
        "예: 흰쌀밥을 먹은 날의 혈당 변화를 설명해 줘",
        key="ai_chat_input",
    )

    if question or auto_clicked:
        q = question.strip() if question else AUTO_QUESTION

        if q:
            # 현재 질문을 추가하기 전에 과거 대화 일부를 가져옴
            previous_messages = st.session_state["chat_messages"][-8:]

            history_text = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: "
                f"{m['content']}"
                for m in previous_messages
            )

            prefix = f"(Selected meal: {meal_id}) " if meal_id else ""

            # 기존 Python 혈당 분석 결과를 사용하는 프롬프트
            data_prompt = build_agent_prompt(
                user_id,
                prefix + q,
                cgm=cgm,
                meals=meals,
            ).replace("NaN", "null")

            # 이전 대화와 현재 질문을 함께 AI에 전달
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

            # 사용자 메시지 표시 및 기록
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

