import os
from dotenv import load_dotenv

load_dotenv()

# Telegram API credentials
API_ID = int(os.getenv('API_ID', 1234567890))
API_HASH = os.getenv('API_HASH', 'Confidential')
BOT_TOKEN = os.getenv('BOT_TOKEN', 'Confidential')

# Model paths
MODEL_PATH = 'models/best_xgboost_model.json'
ENCODERS_PATH = 'models/label_encoders.joblib'

# Bot settings
SESSION_NAME = 'mark_predictor_bot'