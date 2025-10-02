import asyncio
import os
import sys
from telethon import TelegramClient, events, Button
from config import API_ID, API_HASH, BOT_TOKEN, MODEL_PATH, ENCODERS_PATH, SESSION_NAME
from predictor import MarkPredictor

class MarkPredictorBot:
    def __init__(self):
        self.client = None
        self.predictor = None
        self.user_sessions = {}
        
    async def initialize(self):
        # Инициализация бота и загрузка модели
        print("🚀 Инициализация бота...")
        
        # Загружаем модель
        try:
            self.predictor = MarkPredictor(MODEL_PATH, ENCODERS_PATH)
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            sys.exit(1)
        
        # Инициализируем клиент Telethon
        self.client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
        
        # Регистрируем обработчики событий
        self._register_handlers()
        
        # Запускаем бота
        await self.client.start(bot_token=BOT_TOKEN)
        print("✅ Бот успешно запущен!")
        
        # Запускаем бесконечный цикл
        await self.client.run_until_disconnected()
    
    def _register_handlers(self):
        # Регистрация обработчиков событий
        
        @self.client.on(events.NewMessage(pattern='/start'))
        async def start_handler(event):
            await self._handle_start(event)
        
        @self.client.on(events.NewMessage)
        async def message_handler(event):
            await self._handle_message(event)
    
    async def _handle_start(self, event):
        # Обработчик команды /start
        user_id = event.sender_id
        self.user_sessions[user_id] = {'step': 0}
        
        # Создаем кнопки для выбора курса
        buttons = []
        for tier in self.predictor.available_tiers:
            buttons.append([Button.text(tier, resize=True)])
        
        await event.reply(
            "🍹 **Привет! Я бот для предсказания балла на основе твоих напитков!**\n\n"
            "Давай начнем! Выбери твой курс:",
            buttons=buttons
        )
        self.user_sessions[user_id]['step'] = 1
    
    async def _handle_message(self, event):
        # Обработчик всех сообщений
        user_id = event.sender_id
        message_text = event.text.strip()
        
        # Игнорируем команды, которые уже обработаны
        if message_text.startswith('/'):
            return
            
        # Если пользователь не в процессе диалога
        if user_id not in self.user_sessions:
            await self._handle_start(event)
            return
        
        session = self.user_sessions[user_id]
        step = session.get('step', 0)
        
        try:
            if step == 1:  # Выбор курса
                await self._handle_tier_selection(event, session, message_text)
            elif step == 2:  # Выбор любимого напитка
                await self._handle_like_drink_selection(event, session, message_text)
            elif step == 3:  # Частота любимого напитка
                await self._handle_like_frequency(event, session, message_text)
            elif step == 4:  # Выбор частого напитка
                await self._handle_often_drink_selection(event, session, message_text)
            elif step == 5:  # Частота частого напитка
                await self._handle_often_frequency(event, session, message_text)
                
        except Exception as e:
            print(f"❌ Ошибка в обработчике: {e}")
            await event.reply("❌ Произошла ошибка. Давай начнем заново - напиши /start")
            self.user_sessions.pop(user_id, None)
    
    async def _handle_tier_selection(self, event, session, message_text):
        # Обработка выбора курса
        if message_text in self.predictor.available_tiers:
            session['tier'] = message_text
            session['step'] = 2
            
            # Создаем кнопки для любимого напитка
            buttons = []
            for drink in self.predictor.available_like_drinks[:10]:
                buttons.append([Button.text(drink, resize=True)])
            buttons.append([Button.text("Показать еще напитки", resize=True)])
            
            await event.reply(
                f"✅ Курс: {message_text}\n\n"
                "🎯 Теперь выбери твой **любимый напиток** из списка:\n"
                f"*Доступно вариантов: {len(self.predictor.available_like_drinks)}*",
                buttons=buttons
            )
        else:
            await event.reply("❌ Пожалуйста, выбери курс из предложенных вариантов")
    
    async def _handle_like_drink_selection(self, event, session, message_text):
        # Обработка выбора любимого напитка
        if message_text == "Показать еще напитки":
            start_idx = session.get('drink_index', 10)
            end_idx = min(start_idx + 10, len(self.predictor.available_like_drinks))
            
            buttons = []
            for drink in self.predictor.available_like_drinks[start_idx:end_idx]:
                buttons.append([Button.text(drink, resize=True)])
            
            if end_idx < len(self.predictor.available_like_drinks):
                buttons.append([Button.text("Показать еще напитки", resize=True)])
            
            session['drink_index'] = end_idx
            await event.reply("🎯 Выбери твой **любимый напиток**:", buttons=buttons)
            
        elif message_text in self.predictor.available_like_drinks:
            session['like_drink_f'] = message_text
            session['step'] = 3
            session.pop('drink_index', None)
            
            await event.reply(
                f"✅ Любимый напиток: {message_text}\n\n"
                "🔢 **Сколько раз в день ты его пьешь?** (в среднем)\n"
                "*Введи число, например: 2*",
                buttons=Button.clear()
            )
        else:
            await event.reply("❌ Пожалуйста, выбери напиток из списка")
    
    async def _handle_like_frequency(self, event, session, message_text):
        # Обработка частоты любимого напитка
        try:
            frequency = int(message_text)
            if frequency < 0 or frequency > 20:
                await event.reply("❌ Пожалуйста, введи разумное число (0-20)")
                return
                
            session['frequency_day'] = frequency
            session['step'] = 4
            
            # Кнопки для напитка, который пьется чаще всего
            buttons = []
            for drink in self.predictor.available_often_drinks[:10]:
                buttons.append([Button.text(drink, resize=True)])
            buttons.append([Button.text("Показать еще напитки", resize=True)])
            
            await event.reply(
                f"✅ Частота: {frequency} раз в день\n\n"
                "☕ Теперь выбери напиток, который ты **пьешь чаще всего**:\n"
                f"*Доступно вариантов: {len(self.predictor.available_often_drinks)}*",
                buttons=buttons
            )
        except ValueError:
            await event.reply("❌ Пожалуйста, введи целое число")
    
    async def _handle_often_drink_selection(self, event, session, message_text):
        # Обработка выбора частого напитка
        if message_text == "Показать еще напитки":
            start_idx = session.get('often_drink_index', 10)
            end_idx = min(start_idx + 10, len(self.predictor.available_often_drinks))
            
            buttons = []
            for drink in self.predictor.available_often_drinks[start_idx:end_idx]:
                buttons.append([Button.text(drink, resize=True)])
            
            if end_idx < len(self.predictor.available_often_drinks):
                buttons.append([Button.text("Показать еще напитки", resize=True)])
            
            session['often_drink_index'] = end_idx
            await event.reply("☕ Выбери напиток, который пьешь **чаще всего**:", buttons=buttons)
            
        elif message_text in self.predictor.available_often_drinks:
            session['often_drink_f'] = message_text
            session['step'] = 5
            session.pop('often_drink_index', None)
            
            await event.reply(
                f"✅ Частый напиток: {message_text}\n\n"
                "🔢 **Сколько раз в день ты его пьешь?** (в среднем)\n"
                "*Введи число, например: 3*",
                buttons=Button.clear()
            )
        else:
            await event.reply("❌ Пожалуйста, выбери напиток из списка")
    
    async def _handle_often_frequency(self, event, session, message_text):
        # Обработка частоты частого напитка и предсказание
        try:
            frequency = int(message_text)
            if frequency < 0 or frequency > 20:
                await event.reply("❌ Пожалуйста, введи разумное число (0-20)")
                return
                
            session['frequency_day_o'] = frequency
            
            # ВСЕ ДАННЫЕ СОБРАНЫ - ДЕЛАЕМ ПРЕДСКАЗАНИЕ
            await event.reply("⏳ Рассчитываю твой балл...")
            
            prediction = self.predictor.predict(
                session['tier'],
                session['frequency_day'],
                session['frequency_day_o'],
                session['like_drink_f'],
                session['often_drink_f']
            )
            
            # Формируем красивый ответ
            result_message = (
                "🎓 **Результат предсказания**\n\n"
                f"📊 **Твой предсказанный балл:** {prediction}\n\n"
                "📝 **Введенные данные:**\n"
                f"• Курс: {session['tier']}\n"
                f"• Любимый напиток: {session['like_drink_f']}\n"
                f"• Частота любимого: {session['frequency_day']} раз/день\n"
                f"• Частый напиток: {session['often_drink_f']}\n"
                f"• Частота частого: {session['frequency_day_o']} раз/день\n\n"
                "🔄 Хочешь сделать еще одно предсказание? Напиши /start"
            )
            
            await event.reply(result_message)
            
            # Очищаем сессию
            self.user_sessions.pop(event.sender_id, None)
            
        except ValueError:
            await event.reply("❌ Пожалуйста, введи целое число")

async def main():
    # Основная функция запуска бота
    bot = MarkPredictorBot()
    await bot.initialize()

if __name__ == '__main__':
    # Исправление для asyncio event loop
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Бот остановлен")