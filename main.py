import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Загрузка датасета
df = pd.read_csv(r'D:\\Python\Projects\HW1\auto-mpg.csv')

# Вывод первых 5-и строк
print(df.head())
print('-' * 110)

"""
mpg - расход топлива
cylinders - кол-во цилиндров
displacement - объем двигателя
horsepower - лошадиные силы
weight - вес
acceleration - ускорение
model-year - год выпуска
"""

# Вывод информации о датасете
print(df.info())
print('-' * 110)

# Проверка пустых значений
print(df.isnull().sum())
print('-' * 110)

# Вывод характеристик датасета
print(df.describe())

# Удаляем строки с пропущенными значениями
df = df.dropna(subset=['horsepower'])

# Предварительная визуализация данных
sns.pairplot(df)
plt.show()
# Данные не имеют нормального распределения (acceleration тоже не имеет согласно теста Шапиро-Уилка)

# Вывод корреляционной матрицы
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Корреляционная матрица')
plt.show()
# Имеется достаточно много зависимостей

# Определение признаков и целевой переменной
X = df.drop(['mpg'], axis=1)
y = df['mpg']

# Стандартизация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Определение и обучение моделей
models = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "ExtraTrees": ExtraTreesRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

# Создание пустого списка для хранения результатов
results = []

# Тестирование моделей и сбор метрик
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
            'Model': name,
            'MSE': mse,
            'MAE': mae,
            'R2 Score': r2
    })
    # Визуализация фактических и предсказанных значений
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Фактические значения')
    plt.ylabel('Предсказанные значения')
    plt.title('Фактические vs Предсказанные значения метода ' + f'{name}')
    plt.show()
    # Визуализация распределения ошибок
    errors = y_test - y_pred
    plt.figure(figsize=(10, 7))
    sns.histplot(errors, bins=50, kde=True)
    plt.xlabel('Ошибка предсказания')
    plt.title('Распределение ошибок предсказания метода ' + f'{name}')
    plt.show()

# Создание DataFrame и вывод результатов
results_df = pd.DataFrame(results)
print(results_df)
