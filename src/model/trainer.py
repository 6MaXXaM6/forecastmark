"""Обучение модели XGBoost для предсказания среднего балла."""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from src.config import (
    DATASET_PATH,
    MODELS_DIR,
    MODEL_PATH,
    MODEL_ALT_PATH,
    ENCODERS_PATH,
    MODEL_INFO_PATH,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    FEATURE_COLUMNS,
    MARK_MIN,
    MARK_MAX,
    PARAM_GRID,
    RANDOM_STATE,
    CV_RANDOM_STATE,
    TEST_SIZE,
    KFOLD_SPLITS,
)
from src.data.preprocessing import clean_categorical_value, clean_numeric_value

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Информация об обученной модели."""
    params: dict[str, Any]
    test_r2: float
    test_mse: float
    features: list[str] = field(default_factory=lambda: FEATURE_COLUMNS)


class ModelTrainer:
    """Обучает XGBoost модель с K-Fold кросс-валидацией."""

    def __init__(self, dataset_path: str = str(DATASET_PATH)) -> None:
        self.dataset_path: str = dataset_path
        self.df: Optional[pd.DataFrame] = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.best_model: Optional[XGBRegressor] = None
        self.best_params: Optional[dict[str, Any]] = None
        self.best_mse: float = float("inf")
        self.best_fold_results: list[dict[str, Any]] = []

    # ----- Загрузка и предобработка -----

    def load_data(self) -> pd.DataFrame:
        """Загружает CSV и выполняет очистку колонок."""
        logger.info("Загрузка данных из %s", self.dataset_path)
        self.df = pd.read_csv(self.dataset_path)

        for col in CATEGORICAL_COLUMNS:
            self.df[col] = self.df[col].apply(clean_categorical_value)

        for col in NUMERIC_COLUMNS:
            self.df[col] = self.df[col].apply(clean_numeric_value)

        logger.info("Размер до удаления пропусков: %d", len(self.df))
        self.df = self.df.dropna(subset=CATEGORICAL_COLUMNS + NUMERIC_COLUMNS)
        logger.info("Размер после удаления пропусков: %d", len(self.df))

        return self.df

    def encode_features(self) -> tuple[pd.DataFrame, pd.Series]:
        """Кодирует категориальные признаки LabelEncoder'ом и возвращает X, y."""
        if self.df is None:
            self.load_data()

        for col in CATEGORICAL_COLUMNS:
            le = LabelEncoder()
            self.df[col + "_encoded"] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le

        self.df[FEATURE_COLUMNS] = self.df[FEATURE_COLUMNS].astype(float)

        return self.df[FEATURE_COLUMNS], self.df[NUMERIC_COLUMNS[0]].astype(float)

    def split_data(self) -> None:
        """Разбивает на обучающую и тестовую выборки."""
        X, y = self.encode_features()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )
        logger.info("Обучающая выборка: %s", self.X_train.shape)
        logger.info("Тестовая выборка: %s", self.X_test.shape)

    # ----- Утилиты -----

    @staticmethod
    def clip_predictions(predictions: np.ndarray) -> np.ndarray:
        """Обрезает предсказания в диапазон [MARK_MIN, MARK_MAX]."""
        return np.clip(predictions, MARK_MIN, MARK_MAX)

    # ----- Кросс-валидация и обучение -----

    def train_with_kfold(self) -> XGBRegressor:
        """Обучает модели на сетке параметров с K-Fold кросс-валидацией."""
        if self.X_train is None:
            self.split_data()

        n_splits = min(KFOLD_SPLITS, len(self.X_train) - 1)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=CV_RANDOM_STATE)

        logger.info("K-Fold кросс-валидация (%d фолдов) на %d наборах параметров",
                     n_splits, len(PARAM_GRID))

        for params_idx, params in enumerate(PARAM_GRID, 1):
            logger.info("[%d/%d] Параметры: %s", params_idx, len(PARAM_GRID), params)

            fold_scores: list[dict[str, Any]] = []
            fold_models: list[XGBRegressor] = []

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.X_train)):
                X_fold_train = self.X_train.iloc[train_idx]
                X_fold_val = self.X_train.iloc[val_idx]
                y_fold_train = self.y_train.iloc[train_idx]
                y_fold_val = self.y_train.iloc[val_idx]

                model = XGBRegressor(**params, random_state=RANDOM_STATE)
                model.fit(X_fold_train, y_fold_train)

                y_pred = model.predict(X_fold_val)
                y_pred_clipped = self.clip_predictions(y_pred)

                fold_r2 = r2_score(y_fold_val, y_pred_clipped)
                fold_mse = mean_squared_error(y_fold_val, y_pred_clipped)

                fold_scores.append({"fold": fold_idx, "r2": fold_r2, "mse": fold_mse})
                fold_models.append(model)

                logger.debug("  Фолд %d: MSE=%.4f, R²=%.4f", fold_idx + 1, fold_mse, fold_r2)

            avg_mse = float(np.mean([s["mse"] for s in fold_scores]))
            avg_r2 = float(np.mean([s["r2"] for s in fold_scores]))
            logger.info("  Среднее: MSE=%.4f | R²=%.4f", avg_mse, avg_r2)

            if avg_mse < self.best_mse:
                self.best_mse = avg_mse
                self.best_params = params
                best_fold_idx = int(np.argmin([s["mse"] for s in fold_scores]))
                self.best_model = fold_models[best_fold_idx]
                self.best_fold_results = fold_scores
                logger.info("  🏆 Новый лучший результат! MSE=%.4f", avg_mse)

        return self.best_model

    # ----- Оценка на тестовой выборке -----

    def evaluate(self) -> ModelInfo:
        """Оценивает лучшую модель на тестовой выборке."""
        if self.best_model is None:
            self.train_with_kfold()

        y_pred = self.best_model.predict(self.X_test)
        y_pred_clipped = self.clip_predictions(y_pred)

        final_mse = mean_squared_error(self.y_test, y_pred_clipped)
        final_r2 = r2_score(self.y_test, y_pred_clipped)

        logger.info("=== Результаты на тестовых данных ===")
        logger.info("MSE:  %.4f", final_mse)
        logger.info("R²:   %.4f", final_r2)
        logger.info("Диапазон предсказаний: %.2f – %.2f",
                     y_pred_clipped.min(), y_pred_clipped.max())

        return ModelInfo(
            params=self.best_params,
            test_r2=float(final_r2),
            test_mse=float(final_mse),
        )

    # ----- Сохранение -----

    def save(self, model_info: ModelInfo) -> None:
        """Сохраняет модель, энкодеры и метаинформацию."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        self.best_model.save_model(str(MODEL_PATH))
        joblib.dump(self.best_model, str(MODEL_ALT_PATH))
        logger.info("Модель сохранена: %s", MODEL_PATH)

        joblib.dump(model_info, str(MODEL_INFO_PATH))
        joblib.dump(self.label_encoders, str(ENCODERS_PATH))

        logger.info("Энкодеры сохранены: %s", ENCODERS_PATH)
        logger.info("Метаданные сохранены: %s", MODEL_INFO_PATH)

    # ----- Полный пайплайн -----

    def run(self) -> None:
        """Запускает полный пайплайн: загрузка → обучение → оценка → сохранение."""
        self.load_data()
        self.split_data()
        self.train_with_kfold()
        info = self.evaluate()
        self.save(info)
        logger.info("Обучение завершено. MSE на тесте: %.4f", info.test_mse)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    trainer = ModelTrainer()
    trainer.run()
