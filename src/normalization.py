"""음식 이름 정규화: 사진 인식 결과나 사용자가 적은 이름 -> 분석용 food 키.

샘플 데이터(meals.csv)의 food 키(white_rice, brown_rice ...)와 같은 키로 맞춰야
같은 음식끼리 평균 상승폭이 합쳐진다. 모르는 이름은 소문자 + 밑줄로만 바꿔 그대로 쓴다.
새 음식이 자주 나오면 ALIASES에 줄만 추가하면 된다.
"""
from __future__ import annotations

import re

ALIASES = {
    "white_rice": ["흰쌀밥", "쌀밥", "흰밥", "밥", "백미밥", "white rice", "steamed rice", "cooked rice"],
    "brown_rice": ["현미밥", "현미", "brown rice"],
    "whole_oats": ["귀리", "귀리밥", "오트밀", "오트", "oats", "oatmeal", "whole oats"],
    "boiled_potato": ["감자", "삶은감자", "찐감자", "potato", "boiled potato"],
    "baked_sweet_potato": ["고구마", "군고구마", "구운고구마", "sweet potato", "baked sweet potato"],
    "banana": ["바나나", "banana"],
    "apple": ["사과", "apple"],
    "orange": ["오렌지", "귤", "orange"],
    "mango": ["망고", "mango"],
    "pineapple": ["파인애플", "pineapple"],
    "strawberry": ["딸기", "strawberry", "strawberries"],
    "sandwich": ["샌드위치", "sandwich"],
    "broccoli": ["브로콜리", "broccoli"],
}


def _squash(text: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(text).strip().lower())


_LOOKUP = {_squash(alias): key for key, names in ALIASES.items() for alias in [key, *names]}


def normalize_food(name: str) -> str:
    """'흰쌀밥' / 'White Rice' -> 'white_rice'. 모르는 이름은 소문자+밑줄 형태로 반환."""
    hit = _LOOKUP.get(_squash(name))
    if hit:
        return hit
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())