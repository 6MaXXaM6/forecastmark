"""Функции предобработки и очистки данных опроса."""

import re
from typing import Any, Optional

import pandas as pd


# Диапазон допустимых значений балла
MARK_MIN: float = 2.0
MARK_MAX: float = 5.0
MARK_DEFAULT: float = 4.0

# Максимально допустимая частота потребления напитков в день
FREQUENCY_MIN: int = 0
FREQUENCY_MAX: int = 20


def clean_numeric_value(value: Any, default: float = MARK_DEFAULT) -> float:
    """Очищает и нормализует числовое значение (балл/частота).

    Args:
        value: Исходное значение (строка, число или NaN).
        default: Значение по умолчанию, если очистка невозможна.

    Returns:
        Число с плавающей точкой в диапазоне [MARK_MIN, MARK_MAX].
    """
    if pd.isna(value) or value == "":
        return default

    value_str = str(value).strip().replace(",", ".")
    cleaned = re.sub(r"[^\d.]", "", value_str)

    try:
        result = float(cleaned)
        return max(MARK_MIN, min(MARK_MAX, result))
    except ValueError:
        return default


def clean_categorical_value(value: Any) -> Optional[str]:
    """Очищает категориальное значение: нижний регистр, обрезка пробелов.

    Args:
        value: Исходное значение.

    Returns:
        Очищенная строка в нижнем регистре или None.
    """
    if pd.isna(value):
        return None
    cleaned = str(value).strip().lower()
    return cleaned if cleaned else None
