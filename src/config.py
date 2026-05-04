"""Конфигурация приложения ForecastMark.

Загружает настройки из переменных окружения через .env файл.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем .env из корня проекта
load_dotenv()

# ---- Пути ----
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

MODEL_PATH: Path = MODELS_DIR / "best_xgboost_model.json"
MODEL_ALT_PATH: Path = MODELS_DIR / "best_xgboost_model.joblib"
ENCODERS_PATH: Path = MODELS_DIR / "label_encoders.joblib"
MODEL_INFO_PATH: Path = MODELS_DIR / "model_info.joblib"
DATASET_PATH: Path = DATA_DIR / "drinks.csv"

# ---- Telegram API ----
API_ID: int = int(os.getenv("API_ID", "0"))
API_HASH: str = os.getenv("API_HASH", "")
BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")

# ---- Бот ----
SESSION_NAME: str = "mark_predictor_bot"

# ---- Модель ----
RANDOM_STATE: int = 52
CV_RANDOM_STATE: int = 42
TEST_SIZE: float = 0.05

# Диапазоны значений
MARK_MIN: float = 2.0
MARK_MAX: float = 5.0
MARK_DEFAULT: float = 4.0
FREQUENCY_MIN: int = 0
FREQUENCY_MAX: int = 20

# K-Fold
KFOLD_SPLITS: int = 5

# Сетка гиперпараметров для подбора модели
PARAM_GRID: list[dict] = [
    {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 4},
    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
    {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 4},
    {"n_estimators": 100, "learning_rate": 0.01, "max_depth": 4},
]

# Категориальные и числовые столбцы
CATEGORICAL_COLUMNS: list[str] = ["tier", "like_drink_f", "often_drink_f"]
NUMERIC_COLUMNS: list[str] = ["mark"]
FEATURE_COLUMNS: list[str] = [
    "tier_encoded",
    "frequency_day",
    "frequency_day_o",
    "like_drink_f_encoded",
    "often_drink_f_encoded",
]
