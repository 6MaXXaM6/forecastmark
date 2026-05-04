"""Предсказание среднего балла по обученной модели XGBoost."""

import logging
from typing import Optional

import pandas as pd
from xgboost import XGBRegressor
import joblib

from src.config import (
    MARK_MIN,
    MARK_MAX,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
)
from src.data.preprocessing import clean_categorical_value, clean_numeric_value

logger = logging.getLogger(__name__)


class MarkPredictor:
    """Класс для предсказания среднего балла на основе 5 признаков.

    Принимает на вход: tier, frequency_day, frequency_day_o, like_drink_f, often_drink_f.
    """

    def __init__(self, model_path: str, encoders_path: str) -> None:
        self.model: XGBRegressor = self._load_model(model_path)
        self.label_encoders: dict = self._load_encoders(encoders_path)
        self._setup_available_options()

    # ----- Загрузка -----

    def _load_model(self, model_path: str) -> XGBRegressor:
        """Загружает модель XGBoost из файла."""
        model = XGBRegressor()
        model.load_model(model_path)
        logger.info("Модель загружена: %s", model_path)
        return model

    def _load_encoders(self, encoders_path: str) -> dict:
        """Загружает LabelEncoder'ы из joblib."""
        encoders = joblib.load(encoders_path)
        logger.info("Энкодеры загружены: %s", encoders_path)
        return encoders

    def _setup_available_options(self) -> None:
        """Извлекает списки доступных значений из энкодеров."""
        self.available_tiers: list[str] = list(self.label_encoders["tier"].classes_)
        self.available_like_drinks: list[str] = list(
            self.label_encoders["like_drink_f"].classes_
        )
        self.available_often_drinks: list[str] = list(
            self.label_encoders["often_drink_f"].classes_
        )
        logger.info(
            "Доступно: %d курсов, %d любимых напитков, %d частых напитков",
            len(self.available_tiers),
            len(self.available_like_drinks),
            len(self.available_often_drinks),
        )

    # ----- Предсказание -----

    def predict(
        self,
        tier: str,
        frequency_day: float,
        frequency_day_o: float,
        like_drink_f: str,
        often_drink_f: str,
    ) -> float:
        """Предсказывает средний балл по 5 признакам.

        Returns:
            Предсказанный балл, округлённый до 2 знаков, в диапазоне [2.0, 5.0].
        """
        # Очистка входных данных
        data = {
            "tier": clean_categorical_value(tier),
            "frequency_day": clean_numeric_value(frequency_day),
            "frequency_day_o": clean_numeric_value(frequency_day_o),
            "like_drink_f": clean_categorical_value(like_drink_f),
            "often_drink_f": clean_categorical_value(often_drink_f),
        }

        new_df = pd.DataFrame([data])

        # Кодирование категориальных признаков
        for col in CATEGORICAL_COLUMNS:
            le = self.label_encoders[col]
            try:
                new_df[col + "_encoded"] = le.transform(new_df[col])
            except ValueError:
                new_df[col + "_encoded"] = 0
                logger.warning(
                    "Неизвестное значение '%s' для признака '%s', заменено на 0",
                    new_df[col].iloc[0], col,
                )

        # Предсказание и клиппинг
        raw_prediction: float = float(self.model.predict(new_df[FEATURE_COLUMNS])[0])
        clipped = max(MARK_MIN, min(MARK_MAX, raw_prediction))
        return round(clipped, 2)
