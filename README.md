# 🍹 ForecastMark

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io/)
[![Telethon](https://img.shields.io/badge/Telethon-1.30+-blueviolet.svg)](https://docs.telethon.dev/)

**Telegram-бот, предсказывающий средний балл студента на основе его предпочтений в напитках.**

> «Скажи мне, что ты пьёшь, и я скажу, как ты учишься» — ML-модель на XGBoost, обученная на реальных данных опроса студентов.

Попробовать бота: [@forecastmark_bot](https://t.me/forecastmark_bot)

---

## Как это работает

1. Пользователь вводит 5 параметров:
   - **Курс** (1–6)
   - **Любимый напиток**
   - **Частота** любимого напитка (раз в день)
   - **Напиток, который пьётся чаще всего**
   - **Частота** этого напитка (раз в день)

2. Бот кодирует категориальные признаки через `LabelEncoder` и прогоняет через обученную **XGBoost-модель**

3. Модель возвращает предсказанный балл в диапазоне **2.0 – 5.0**

---

## Структура проекта

```
forecastmark/
├── src/                        # Исходный код
│   ├── config.py               # Конфигурация (пути, константы)
│   ├── data/
│   │   └── preprocessing.py    # Очистка и нормализация данных
│   ├── model/
│   │   ├── trainer.py          # Обучение модели (класс ModelTrainer)
│   │   └── predictor.py        # Предсказание (класс MarkPredictor)
│   └── bot/
│       ├── handlers.py         # Обработчики сообщений бота
│       └── bot.py              # Точка входа бота
├── notebooks/
│   └── What_drink_cmc.ipynb    # Исследовательский ноутбук
├── data/
│   └── drinks.csv              # Датасет (данные опроса)
├── models/                     # Сохранённые модели и энкодеры
├── requirements.txt            # Python-зависимости
└── README.md
```

---

## Быстрый старт

### 1. Клонирование и установка

```bash
git clone https://github.com/<your-username>/forecastmark.git
cd forecastmark
pip install -r requirements.txt
```

### 2. Настройка `.env`

Создай файл `.env` в корне проекта:

```env
API_ID=12345678
API_HASH=your_api_hash_here
BOT_TOKEN=your_bot_token_here
```

Для получения `API_ID` и `API_HASH` зайди на [my.telegram.org](https://my.telegram.org).
`BOT_TOKEN` получи у [@BotFather](https://t.me/BotFather).

### 3. Обучение модели (опционально)

Если хочешь переобучить модель на своих данных:

```bash
python -m src.model.trainer
```

Модель, энкодеры и метаданные сохранятся в `models/`.

### 4. Запуск бота

```bash
python -m src.bot.bot
```

---

## Модель

| Параметр | Значение |
|----------|----------|
| Алгоритм | XGBoost Regressor |
| Кросс-валидация | K-Fold (5 фолдов) |
| Grid Search | Сетка из 4 комбинаций гиперпараметров |
| Метрика | MSE + R² |
| Диапазон выхода | 2.0 – 5.0 (клиппинг) |

---

## Датасет

Данные собраны через Google Forms — ссылка на опрос раскидывалась в чаты всех курсов. Каждая строка — уникальный ответ респондента.

Колонки:
- `tier` — курс (строка)
- `like_drink_f` — любимый напиток (категория)
- `frequency_day` — частота любимого (раз/день)
- `often_drink_f` — самый частый напиток (категория)
- `frequency_day_o` — частота частого (раз/день)
- `mark` — реальный средний балл (целевая переменная)

---

## 🛠 Технологии

- **Python 3.10+**
- **XGBoost** — градиентный бустинг
- **scikit-learn** — предобработка, кросс-валидация
- **Telethon** — асинхронный клиент Telegram API

---

## Дисклеймер

Этот проект создан в развлекательных целях и **не претендует на научность**. Корреляция между напитками и успеваемостью не доказана я даже ноутбук с анализом решил не делать (но кто знает?))).