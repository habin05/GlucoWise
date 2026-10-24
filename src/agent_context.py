"""Step 5: turn pre-computed glucose analysis into the text the Foundry agent receives.

The AI must only EXPLAIN these numbers; it must not calculate or invent any. All data is synthetic.

    from src.agent_context import build_agent_prompt
    prompt = build_agent_prompt("A", "How did white rice compare with brown rice?")
    response = openai_client.responses.create(input=prompt)
"""
from __future__ import annotations

import json

try:  # works as `src.agent_context` (Streamlit) and as a plain script from src/
    from .glucose_analysis import analyze_user_patterns, get_meal_history
except ImportError:  # pragma: no cover
    from glucose_analysis import analyze_user_patterns, get_meal_history

NOTICE = (
    "All values are SIMULATED demo data, not real measurements. Report only what the data shows; "
    "do not give medical advice or claim that a food causes a glucose response."
)


def build_agent_context(user_id: str, cgm=None, meals=None) -> dict:
    """Everything the agent may use for one user: per-food summary plus every past meal."""
    summary = analyze_user_patterns(user_id, cgm=cgm, meals=meals)
    history = get_meal_history(user_id, cgm=cgm, meals=meals)
    return {
        "user_id": user_id,
        "data_type": summary["data_type"],
        "unit": summary["unit"],
        "definition": summary["definition"],
        "notice": NOTICE,
        "food_summary": summary["patterns"],
        "meals": [
            {
                "meal_id": h["meal_id"],
                "date_time": h["timestamp"][:16].replace("T", " "),
                "food": h["food"],
                "description": h["description"],
                "primary_portion_g": h["primary_portion_g"],
                "baseline": h["baseline"],
                "peak": h["peak"],
                "rise": h["rise"],
            }
            for h in history
        ],
    }


def build_agent_prompt(user_id: str, question: str, cgm=None, meals=None) -> str:
    """One string to pass as `input=` to the agent: data first, then the user's question."""
    context = build_agent_context(user_id, cgm=cgm, meals=meals)
    return (
        "Pre-computed analysis (JSON). Use only these numbers.\n"
        + json.dumps(context, indent=1)
        + "\n\nUser question: "
        + question
    )


if __name__ == "__main__":
    print(build_agent_prompt("A", "Show my white rice meals."))
