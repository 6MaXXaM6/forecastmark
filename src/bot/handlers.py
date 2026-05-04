"""Обработчики сообщений Telegram-бота."""

import logging
from typing import Dict, Any, Optional

from telethon import events, Button
from telethon.tl.custom import Message

from src.model.predictor import MarkPredictor

logger = logging.getLogger(__name__)

# Максимальное количество кнопок «Показать ещё» за раз
BUTTONS_PER_PAGE: int = 10
FREQUENCY_MIN: int = 0
FREQUENCY_MAX: int = 20

# Хранилище сессий пользователей: {user_id: dict}
user_sessions: Dict[int, Dict[str, Any]] = {}

# Ссылка на предиктор (устанавливается из bot.py)
predictor: Optional[MarkPredictor] = None


def init_handlers(p: MarkPredictor) -> None:
    """Инициализирует обработчики ссылкой на предиктор."""
    global predictor
    predictor = p


# ----- Хелперы -----

def _build_drink_buttons(
    drinks: list[str],
    start: int = 0,
    label: str = "Показать ещё напитки",
) -> list:
    """Создаёт кнопки с напитками + кнопку «Показать ещё»."""
    chunk = drinks[start : start + BUTTONS_PER_PAGE]
    buttons = [[Button.text(d, resize=True)] for d in chunk]
    if start + BUTTONS_PER_PAGE < len(drinks):
        buttons.append([Button.text(label, resize=True)])
    return buttons


def _show_more_drinks(
    drinks: list[str],
    session: dict,
    index_key: str,
) -> list:
    """Логика пагинации «Показать ещё» для напитков.

    Возвращает список кнопок для следующей страницы и обновляет сессию.
    """
    start = session.get(index_key, BUTTONS_PER_PAGE)
    buttons = _build_drink_buttons(drinks, start)
    new_page = start + BUTTONS_PER_PAGE
    session[index_key] = new_page if new_page < len(drinks) else len(drinks)
    return buttons


# ----- Обработчик /start -----

async def handle_start(event: events.NewMessage.Event) -> None:
    """Обрабатывает команду /start — начинает диалог."""
    user_id = event.sender_id
    user_sessions[user_id] = {"step": 1}

    buttons = [[Button.text(t, resize=True)] for t in predictor.available_tiers]

    await event.reply(
        "🍹 **Привет! Я бот для предсказания балла на основе твоих напитков!**\n\n"
        "Давай начнём! Выбери твой курс:",
        buttons=buttons,
    )
    logger.info("Пользователь %d начал диалог", user_id)


# ----- Главный диспетчер -----

async def handle_message(event: events.NewMessage.Event) -> None:
    """Диспетчеризует сообщение по текущему шагу сессии пользователя."""
    user_id = event.sender_id
    text = event.text.strip()

    # Игнорируем команды (они обрабатываются отдельно)
    if text.startswith("/"):
        return

    if user_id not in user_sessions:
        await handle_start(event)
        return

    session = user_sessions[user_id]
    step = session.get("step", 0)

    try:
        if step == 1:
            await _handle_tier(event, session, text)
        elif step == 2:
            await _handle_like_drink(event, session, text)
        elif step == 3:
            await _handle_like_frequency(event, session, text)
        elif step == 4:
            await _handle_often_drink(event, session, text)
        elif step == 5:
            await _handle_often_frequency(event, session, text)
    except Exception:
        logger.exception("Ошибка обработки для пользователя %d", user_id)
        await event.reply("❌ Произошла ошибка. Давай начнём заново — напиши /start")
        user_sessions.pop(user_id, None)


# ----- Шаг 1: выбор курса -----

async def _handle_tier(
    event: events.NewMessage.Event, session: dict, text: str
) -> None:
    if text not in predictor.available_tiers:
        await event.reply("❌ Пожалуйста, выбери курс из предложенных вариантов")
        return

    session["tier"] = text
    session["step"] = 2

    buttons = _build_drink_buttons(predictor.available_like_drinks, 0)
    await event.reply(
        f"✅ Курс: {text}\n\n"
        "🎯 Теперь выбери твой **любимый напиток** из списка:\n"
        f"_Доступно вариантов: {len(predictor.available_like_drinks)}_",
        buttons=buttons,
    )


# ----- Шаг 2: любимый напиток -----

async def _handle_like_drink(
    event: events.NewMessage.Event, session: dict, text: str
) -> None:
    if text == "Показать ещё напитки":
        buttons = _show_more_drinks(
            predictor.available_like_drinks,
            session,
            "drink_page",
        )
        await event.reply("🎯 Выбери твой **любимый напиток**:", buttons=buttons)
        return

    if text not in predictor.available_like_drinks:
        await event.reply("❌ Пожалуйста, выбери напиток из списка")
        return

    session["like_drink_f"] = text
    session["step"] = 3
    session.pop("drink_page", None)

    await event.reply(
        f"✅ Любимый напиток: {text}\n\n"
        "🔢 **Сколько раз в день ты его пьёшь?** (в среднем)\n"
        "_Введи число, например: 2_",
        buttons=Button.clear(),
    )


# ----- Шаг 3: частота любимого -----

async def _handle_like_frequency(
    event: events.NewMessage.Event, session: dict, text: str
) -> None:
    try:
        freq = int(text)
        if not (FREQUENCY_MIN <= freq <= FREQUENCY_MAX):
            await event.reply(f"❌ Пожалуйста, введи разумное число ({FREQUENCY_MIN}–{FREQUENCY_MAX})")
            return

        session["frequency_day"] = freq
        session["step"] = 4

        buttons = _build_drink_buttons(predictor.available_often_drinks, 0)
        await event.reply(
            f"✅ Частота: {freq} раз/день\n\n"
            "☕ Теперь выбери напиток, который ты **пьёшь чаще всего**:\n"
            f"_Доступно вариантов: {len(predictor.available_often_drinks)}_",
            buttons=buttons,
        )
    except ValueError:
        await event.reply("❌ Пожалуйста, введи целое число")


# ----- Шаг 4: частый напиток -----

async def _handle_often_drink(
    event: events.NewMessage.Event, session: dict, text: str
) -> None:
    if text == "Показать ещё напитки":
        buttons = _show_more_drinks(
            predictor.available_often_drinks,
            session,
            "often_drink_page",
        )
        await event.reply(
            "☕ Выбери напиток, который пьёшь **чаще всего**:", buttons=buttons
        )
        return

    if text not in predictor.available_often_drinks:
        await event.reply("❌ Пожалуйста, выбери напиток из списка")
        return

    session["often_drink_f"] = text
    session["step"] = 5
    session.pop("often_drink_page", None)

    await event.reply(
        f"✅ Частый напиток: {text}\n\n"
        "🔢 **Сколько раз в день ты его пьёшь?** (в среднем)\n"
        "_Введи число, например: 3_",
        buttons=Button.clear(),
    )


# ----- Шаг 5: частота частого → предсказание -----

async def _handle_often_frequency(
    event: events.NewMessage.Event, session: dict, text: str
) -> None:
    try:
        freq = int(text)
        if not (FREQUENCY_MIN <= freq <= FREQUENCY_MAX):
            await event.reply(f"❌ Пожалуйста, введи разумное число ({FREQUENCY_MIN}–{FREQUENCY_MAX})")
            return

        session["frequency_day_o"] = freq

        await event.reply("⏳ Рассчитываю твой балл...")

        prediction = predictor.predict(
            tier=session["tier"],
            frequency_day=session["frequency_day"],
            frequency_day_o=session["frequency_day_o"],
            like_drink_f=session["like_drink_f"],
            often_drink_f=session["often_drink_f"],
        )

        result = (
            "🎓 **Результат предсказания**\n\n"
            f"📊 **Твой предсказанный балл:** {prediction}\n\n"
            "📝 **Введённые данные:**\n"
            f"• Курс: {session['tier']}\n"
            f"• Любимый напиток: {session['like_drink_f']}\n"
            f"• Частота любимого: {session['frequency_day']} раз/день\n"
            f"• Частый напиток: {session['often_drink_f']}\n"
            f"• Частота частого: {session['frequency_day_o']} раз/день\n\n"
            "🔄 Хочешь сделать ещё одно предсказание? Напиши /start"
        )

        await event.reply(result)
        logger.info("Предсказание для пользователя %d: %.2f", event.sender_id, prediction)

    except ValueError:
        await event.reply("❌ Пожалуйста, введи целое число")
    finally:
        user_sessions.pop(event.sender_id, None)
