
"""Food recognition and normalization using Microsoft Foundry."""

import base64
import json
import os
import re

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from normalization import normalize_food


ENDPOINT = os.getenv(
    "FOUNDRY_ENDPOINT",
    "https://whb050704-4696-resource.services.ai.azure.com/api/projects/whb050704-4696",
)

AGENT_NAME = os.getenv(
    "FOOD_NORMALIZER_AGENT",
    "GlucoWise-Food-Normalizer",
)


def recognize_foods(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
) -> list[str]:
    """Send a food photo to Foundry and return normalized food names.

    Example return value:
        ["sandwich", "cherry_tomato"]

    The Foundry agent must support image inputs and return JSON.
    """

    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError(f"Unsupported image type: {mime_type}")

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{mime_type};base64,{image_base64}"

    request = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": """
Identify all clearly visible foods in this image.
Normalize each food name to a short, lowercase,
English snake_case name.

Examples:
- Cherry Tomato -> cherry_tomato
- White Rice -> white_rice
- Sandwich -> sandwich

Return ONLY valid JSON in this format:
{"foods": ["sandwich", "cherry_tomato"]}

Do not guess hidden ingredients or foods that are
not clearly visible. Do not estimate portion sizes.
""",
                },
                {
                    "type": "input_image",
                    "image_url": image_url,
                },
            ],
        }
    ]

    with DefaultAzureCredential() as credential:
        with AIProjectClient(
            endpoint=ENDPOINT,
            credential=credential,
            allow_preview=True,
        ) as project:
            client = project.get_openai_client(
                agent_name=AGENT_NAME
            )
            response = client.responses.create(input=request)

    output = response.output_text.strip()

    # Handle responses wrapped in Markdown code fences.
    output = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        output,
        flags=re.IGNORECASE,
    ).strip()

    # Parse JSON directly, or extract a JSON object/array.
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", output, re.DOTALL)
        if not match:
            raise ValueError(
                f"The food agent did not return valid JSON: {output}"
            )
        data = json.loads(match.group(0))

    if isinstance(data, dict):
        foods = data.get("foods", [])
    elif isinstance(data, list):
        foods = data
    else:
        raise ValueError("Unexpected JSON format from the food agent.")

    if not isinstance(foods, list):
        raise ValueError("The 'foods' field must be a JSON list.")

    # Support both strings and structured food objects.
    normalized_foods = []
    for food in foods:
        if isinstance(food, str):
            name = food.strip().lower()
        elif isinstance(food, dict):
            name = food.get("normalized_name", "")
            if isinstance(name, str):
                name = name.strip().lower()
            else:
                name = ""
        else:
            name = ""
        if name and name not in normalized_foods:
            normalized_foods.append(name)
        # (위쪽 파싱 코드는 그대로)
    result = []
    for food in normalized_foods:
        name = normalize_food(food)
        if name and name not in result:
            result.append(name)
    return result


def normalize_food_text(text: str) -> str:
    """Normalize user-edited food text into an English snake_case label."""
    if not text or not text.strip():
        raise ValueError("Food text cannot be empty.")

    request = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"""
Normalize the following food name or meal description.

Input may be in any language.
Return a concise, standard English label using lowercase
snake_case, such as white_rice or kimchi_fried_rice.

Rules:
- Translate non-English food names into English.
- Use consistent names for equivalent foods.
- Preserve meaningful ingredients and preparation methods.
- If multiple foods are listed, preserve the meaningful
  food information in one concise label.
- Do not invent ingredients.
- Return ONLY the normalized label, with no explanation.

Input: {text}
""",
                }
            ],
        }
    ]

    with DefaultAzureCredential() as credential:
        with AIProjectClient(
            endpoint=ENDPOINT,
            credential=credential,
            allow_preview=True,
        ) as project:
            client = project.get_openai_client(agent_name=AGENT_NAME)
            response = client.responses.create(input=request)

    output = response.output_text.strip()
    output = re.sub(
        r"^```(?:text)?\s*|\s*```$", "", output, flags=re.IGNORECASE
    ).strip()

    label = re.sub(r"[^a-zA-Z0-9]+", "_", output).strip("_").lower()
    if not label:
        raise ValueError("The AI could not return a valid English food label.")

    # AI 결과도 샘플 데이터 키(white_rice 등)와 맞추기 위해 alias 사전을 한 번 더 통과
    return normalize_food(label)
