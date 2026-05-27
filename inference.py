"""
inference.py — скрипт для предсказания сердечно-сосудистых заболеваний
Модель: HeartDiseaseNN (PyTorch, полносвязная нейронная сеть)
Accuracy: 89.75% | ROC-AUC: 96.20%

Использование:
    # Предсказание для одного пациента (пример)
    python inference.py --mode single

    # Предсказание для CSV файла
    python inference.py --mode csv --input data/heart_test.csv --output predictions.csv
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import argparse
import os


# ============================================================
# АРХИТЕКТУРА МОДЕЛИ (должна точно совпадать с обучением)
# ============================================================
class HeartDiseaseNN(nn.Module):
    def __init__(self, input_dim):
        super(HeartDiseaseNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze()


# ============================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================
def load_model(
    model_path='best_model.pth',
    scaler_path='scaler.pkl',
    columns_path='feature_columns.pkl'
):
    """Загружает веса модели, scaler и список признаков."""
    for path in [model_path, scaler_path, columns_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не найден: {path}")

    with open(columns_path, 'rb') as f:
        feature_columns = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeartDiseaseNN(input_dim=len(feature_columns)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Модель загружена | Устройство: {device} | Признаков: {len(feature_columns)}")
    return model, scaler, feature_columns, device


# ============================================================
# ПРЕДОБРАБОТКА (повторяет логику из analysis.ipynb)
# ============================================================
def preprocess(df, scaler, feature_columns):
    """
    Предобрабатывает входной датафрейм:
    1. Удаляет ID
    2. Округляет chest и фильтрует невалидные значения
    3. One-Hot кодирование категориальных переменных
    4. Выравнивает колонки под обученную модель
    5. Масштабирует числовые признаки
    """
    df = df.copy()

    # Удаляем ID если есть
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    # Округляем chest
    df['chest'] = df['chest'].round().astype(int)

    # Фильтрация невалидных значений
    df = df[df['chest'].isin([1, 2, 3, 4])]
    df = df[df['thal'].isin([3, 6, 7])]
    df = df[df['slope'].isin([1, 2, 3])]
    df = df[df['number_of_major_vessels'].isin([0, 1, 2, 3])]

    # One-Hot кодирование категориальных переменных
    cat_cols = [
        'sex', 'chest', 'fasting_blood_sugar',
        'resting_electrocardiographic_results',
        'exercise_induced_angina', 'slope',
        'number_of_major_vessels', 'thal'
    ]
    df = pd.get_dummies(df, columns=cat_cols)

    # Выравниваем колонки под обученную модель
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Масштабирование числовых признаков
    num_cols = [
        'age', 'resting_blood_pressure', 'serum_cholestoral',
        'maximum_heart_rate_achieved', 'oldpeak'
    ]
    df[num_cols] = scaler.transform(df[num_cols])

    return df.astype(float)


# ============================================================
# ИНФЕРЕНС
# ============================================================
def predict(df, model, scaler, feature_columns, device, threshold=0.5):
    """Возвращает предсказания и вероятности для датафрейма."""
    processed = preprocess(df, scaler, feature_columns)
    X_tensor = torch.FloatTensor(processed.values).to(device)

    model.eval()
    with torch.no_grad():
        probas = model(X_tensor).cpu().numpy()

    if probas.ndim == 0:
        probas = np.array([probas])

    predictions = (probas >= threshold).astype(int)
    return predictions, probas


# ============================================================
# РЕЖИМ 1: ОДИН ПАЦИЕНТ
# ============================================================
def predict_single(patient_data: dict):
    """
    Предсказание для одного пациента.

    Пример данных пациента:
        age                                  — возраст (float)
        sex                                  — пол: 0=женщина, 1=мужчина
        chest                                — тип боли в груди: 1-4
        resting_blood_pressure               — давление в покое
        serum_cholestoral                    — холестерин (mg/dl)
        fasting_blood_sugar                  — глюкоза натощак > 120: 0/1
        resting_electrocardiographic_results — ЭКГ в покое: 0/1/2
        maximum_heart_rate_achieved          — макс. ЧСС
        exercise_induced_angina              — стенокардия при нагрузке: 0/1
        oldpeak                              — депрессия ST
        slope                                — наклон ST: 1/2/3
        number_of_major_vessels              — кол-во сосудов: 0-3
        thal                                 — тип дефекта: 3/6/7
    """
    model, scaler, feature_columns, device = load_model()
    df = pd.DataFrame([patient_data])
    predictions, probas = predict(df, model, scaler, feature_columns, device)

    print("\n" + "=" * 40)
    print("       РЕЗУЛЬТАТ ПРЕДСКАЗАНИЯ")
    print("=" * 40)
    print(f"  Вероятность болезни: {probas[0]:.4f} ({probas[0]*100:.1f}%)")
    if predictions[0] == 1:
        print("  Предсказание:        ⚠️  БОЛЕЗНЬ ОБНАРУЖЕНА")
    else:
        print("  Предсказание:        ✅  ЗДОРОВ")
    print("=" * 40)
    return int(predictions[0]), float(probas[0])


# ============================================================
# РЕЖИМ 2: CSV ФАЙЛ
# ============================================================
def predict_csv(input_path, output_path='predictions.csv'):
    """Предсказание для всех записей в CSV файле."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    model, scaler, feature_columns, device = load_model()

    print(f"\n📂 Чтение данных: {input_path}")
    df = pd.read_csv(input_path)
    original_ids = df['ID'].values if 'ID' in df.columns else np.arange(len(df))
    print(f"   Загружено записей: {len(df)}")

    predictions, probas = predict(df, model, scaler, feature_columns, device)

    result = pd.DataFrame({
        'ID': original_ids[:len(predictions)],
        'prediction': predictions,
        'probability': np.round(probas, 4)
    })
    result.to_csv(output_path, index=False)

    print(f"\nРезультаты:")
    print(f"   Здоровых : {(predictions == 0).sum():,}")
    print(f"   Больных  : {(predictions == 1).sum():,}")
    print(f"   Сохранено: {output_path}")
    return result


# ============================================================
# ТОЧКА ВХОДА
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Heart Disease Detection — инференс нейронной сети'
    )
    parser.add_argument(
        '--mode', type=str, choices=['single', 'csv'], default='single',
        help='Режим запуска: single (один пациент) или csv (файл)'
    )
    parser.add_argument(
        '--input', type=str, default=None,
        help='Путь к входному CSV файлу (для режима csv)'
    )
    parser.add_argument(
        '--output', type=str, default='predictions.csv',
        help='Путь для сохранения предсказаний (для режима csv)'
    )
    args = parser.parse_args()

    if args.mode == 'csv':
        if args.input is None:
            print("Укажите путь к файлу: --input data/heart_test.csv")
        else:
            predict_csv(args.input, args.output)

    else:
        # Пример пациента с высоким риском
        patient = {
            'age': 63.0,
            'sex': 1,
            'chest': 4,
            'resting_blood_pressure': 145.0,
            'serum_cholestoral': 233.0,
            'fasting_blood_sugar': 1,
            'resting_electrocardiographic_results': 2,
            'maximum_heart_rate_achieved': 150.0,
            'exercise_induced_angina': 0,
            'oldpeak': 2.3,
            'slope': 3,
            'number_of_major_vessels': 0,
            'thal': 6
        }
        print("\nДанные пациента:")
        for k, v in patient.items():
            print(f"   {k}: {v}")
        predict_single(patient)
