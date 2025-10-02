import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import re
import joblib

class MarkPredictor:
    def __init__(self, model_path, encoders_path):
        self.model = self._load_model(model_path)
        self.label_encoders = self._load_encoders(encoders_path)
        self._setup_available_options()
    
    def _load_model(self, model_path):
        # Загружает модель XGBoost
        model = XGBRegressor()
        model.load_model(model_path)
        print(f"Модель загружена из {model_path}")
        return model
    
    def _load_encoders(self, encoders_path):
        # Загружает LabelEncoders
        encoders = joblib.load(encoders_path)
        print(f"Энкодеры загружены из {encoders_path}")
        return encoders
    
    def _setup_available_options(self):
        # Извлекает доступные варианты из энкодеров
        self.available_tiers = list(self.label_encoders['tier'].classes_)
        self.available_like_drinks = list(self.label_encoders['like_drink_f'].classes_)
        self.available_often_drinks = list(self.label_encoders['often_drink_f'].classes_)
    
    @staticmethod
    def clean_numeric_value(value, default=4.0):
        # Очищает числовые значения
        if pd.isna(value) or value == '':
            return default
        value_str = str(value).strip().replace(',', '.')
        cleaned = re.sub(r'[^\d\.]', '', value_str)
        try:
            result = float(cleaned)
            return max(2.0, min(5.0, result))
        except ValueError:
            return default
    
    @staticmethod
    def clean_categorical_value(value):
        # Очищает категориальные значения
        if pd.isna(value):
            return None
        return str(value).strip().lower()
    
    def predict(self, tier, frequency_day, frequency_day_o, like_drink_f, often_drink_f):
        # Основная функция предсказания

        # Очищаем входные данные
        tier_clean = self.clean_categorical_value(tier)
        frequency_day_clean = self.clean_numeric_value(frequency_day)
        frequency_day_o_clean = self.clean_numeric_value(frequency_day_o)
        like_drink_f_clean = self.clean_categorical_value(like_drink_f)
        often_drink_f_clean = self.clean_categorical_value(often_drink_f)
        
        # Создаем DataFrame для новых данных
        new_data = pd.DataFrame({
            'tier': [tier_clean],
            'frequency_day': [frequency_day_clean],
            'frequency_day_o': [frequency_day_o_clean],
            'like_drink_f': [like_drink_f_clean],
            'often_drink_f': [often_drink_f_clean]
        })
        
        categorical_columns = ['tier', 'like_drink_f', 'often_drink_f']
        
        # Кодируем категориальные признаки
        for col in categorical_columns:
            le = self.label_encoders[col]
            try:
                new_data[col + '_encoded'] = le.transform(new_data[col])
            except ValueError:
                new_data[col + '_encoded'] = 0
                print(f"Новое значение '{new_data[col].iloc[0]}' для признака '{col}'")
        
        # Используем закодированные признаки
        features = ['tier_encoded', 'frequency_day', 'frequency_day_o', 
                   'like_drink_f_encoded', 'often_drink_f_encoded']
        features_new = new_data[features]
        
        # Предсказываем и ограничиваем результат
        prediction = self.model.predict(features_new)[0]
        clipped_prediction = max(2.0, min(5.0, prediction))
        
        return round(clipped_prediction, 2)