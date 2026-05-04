"""Точка входа Telegram-бота ForecastMark."""

import asyncio
import logging
import sys

from telethon import TelegramClient, events

from src.config import API_ID, API_HASH, BOT_TOKEN, SESSION_NAME, MODEL_PATH, ENCODERS_PATH
from src.model.predictor import MarkPredictor
from src.bot.handlers import init_handlers, handle_start, handle_message

logger = logging.getLogger(__name__)


async def main() -> None:
    """Инициализирует и запускает Telegram-бота."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("🚀 Инициализация бота ForecastMark...")

    # Загружаем модель
    try:
        predictor = MarkPredictor(
            model_path=str(MODEL_PATH),
            encoders_path=str(ENCODERS_PATH),
        )
    except Exception:
        logger.exception("❌ Ошибка загрузки модели")
        sys.exit(1)

    # Пробрасываем предиктор в обработчики
    init_handlers(predictor)

    # Инициализируем клиент Telethon
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

    # Регистрируем обработчики
    client.on(events.NewMessage(pattern="/start"))(handle_start)
    client.on(events.NewMessage())(handle_message)

    # Запускаем
    await client.start(bot_token=BOT_TOKEN)
    logger.info("✅ Бот успешно запущен!")

    await client.run_until_disconnected()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Бот остановлен")
