import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import re

# Загрузка данных
df = pd.read_csv('drinks.csv')

# Функция для очистки балла
def clean_numeric_value(value, default=4.0):
    if pd.isna(value) or value == '':
        return default
    value_str = str(value).strip().replace(',', '.')
    cleaned = re.sub(r'[^\d\.]', '', value_str)
    
    try:
        result = float(cleaned)
        # Ограничиваем значение в диапазоне от 2 до 5 от умников
        return max(2.0, min(5.0, result))
    except ValueError:
        return default

# Функция для очистки категориальных значений
def clean_categorical_value(value):
    if pd.isna(value):
        return None
    cleaned = str(value).strip().lower()
    return cleaned if cleaned != '' else None

# Очищаем категориальные столбцы
categorical_columns = ['tier', 'like_drink_f', 'often_drink_f']
for col in categorical_columns:
    df[col] = df[col].apply(clean_categorical_value)

# Очищаем числовые столбцы
numeric_columns = ['mark']
for col in numeric_columns:
    df[col] = df[col].apply(clean_numeric_value)

print(f"Размер данных до удаления пропусков: {len(df)}")
# Удаляем строки, где пропущены категориальные признаки или целевая переменная
df = df.dropna(subset=categorical_columns + numeric_columns)

print(f"Размер данных после удаления пропусков: {len(df)}")

# Векторизация категориальных признаков с помощью LabelEncoder
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# Определение признаков и целевой переменной
features = ['tier_encoded', 'frequency_day', 'frequency_day_o', 'like_drink_f_encoded', 'often_drink_f_encoded']
X = df[features]
y = df['mark']

# Проверяем типы данных
print(f"Типы данных в X:")
print(X.dtypes)
print(f"Тип данных в y: {y.dtype}")

# Разделение на обучающую и тестовую выборки чтобы отследить как обучилась скорее по фолдам поймём чисто символически оставляю пару строк
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=52)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
    

# K-FOLD КРОСС-ВАЛИДАЦИЯ ДЛЯ ПОИСКА ЛУЧШЕЙ МОДЕЛИ
def clip_predictions(predictions):
    return np.clip(predictions, 2.0, 5.0)

# Настройки K-Fold
n_splits = min(5, len(X_train) - 1)  # Адаптируем количество фолдов под объем данных
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Параметры для перебора
param_combinations = [
    {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 4},
    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6},
    {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 4},
    {'n_estimators': 5000, 'learning_rate': 0.1, 'max_depth': 20}, # по приколу
]

best_model = None
best_mse = np.inf
best_params = None
best_fold_results = []

print(f"\nНачинаем K-Fold кросс-валидацию ({n_splits} фолдов)...")

for param_idx, params in enumerate(param_combinations):
    print(f"\nТестируем комбинацию параметров {param_idx + 1}/{len(param_combinations)}: {params}")

    fold_scores = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = XGBRegressor(**params, random_state=52)
        model.fit(X_fold_train, y_fold_train)
        
        y_fold_pred = model.predict(X_fold_val)
        y_fold_pred_clipped = clip_predictions(y_fold_pred)
        
        fold_r2 = r2_score(y_fold_val, y_fold_pred_clipped)
        fold_mse = mean_squared_error(y_fold_val, y_fold_pred_clipped)
        
        fold_scores.append({'fold': fold, 'r2': fold_r2, 'mse': fold_mse})
        fold_models.append(model)
        
        print(f"Фолд {fold + 1}: MSE = {fold_mse:.4f}, R2 = {fold_r2:.4f}")
    
    avg_mse = np.mean([score['mse'] for score in fold_scores])
    avg_r2 = np.mean([score['r2'] for score in fold_scores])
    
    print(f"Средние по фолдам: MSE = {avg_mse:.4f}, R2 = {avg_r2:.4f}")
    
    if avg_mse < best_mse:
        best_mse = avg_mse
        best_params = params
        best_fold_idx = np.argmin([score['mse'] for score in fold_scores])
        best_model = fold_models[best_fold_idx]
        best_fold_results = fold_scores
        print(f"НОВАЯ ЛУЧШАЯ МОДЕЛЬ! MSE = {avg_mse:.4f}")

print("\nФИНАЛЬНАЯ ОЦЕНКА ЛУЧШЕЙ МОДЕЛИ")
print(f"Лучшие параметры: {best_params}")
print(f"Средний MSE на кросс-валидации: {best_mse:.4f}")

y_pred = best_model.predict(X_test)
y_pred_clipped = clip_predictions(y_pred)

final_mse = mean_squared_error(y_test, y_pred_clipped)
final_r2 = r2_score(y_test, y_pred_clipped)

print(f"\nРезультаты на тестовых данных:")
print(f"Среднеквадратичная ошибка (MSE): {final_mse:.4f}")
print(f"Коэффициент детерминации (R2): {final_r2:.4f}")
print(f"Диапазон предсказаний: от {y_pred_clipped.min():.2f} до {y_pred_clipped.max():.2f}")

# Сохраняем модель и энкодеры
best_model.save_model('best_xgboost_model.json')
joblib.dump(best_model, 'best_xgboost_model.joblib')

model_info = {
    'best_params': best_params,
    'test_r2': float(final_r2),
    'test_mse': float(final_mse),
    'features': features,
}

joblib.dump(model_info, 'model_info.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')

print("\nМодель успешно сохранена!")
print("best_xgboost_model.json - основная модель")
print("best_xgboost_model.joblib - модель в joblib формате")
print("model_info.joblib - информация о модели")
print("label_encoders.joblib - кодировщики категориальных признаков")

print(f"Процесс завершен! Лучшая модель сохранена с MSE = {final_mse:.4f} на тестовых данных")