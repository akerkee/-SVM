# Қажетті кітапханаларды импорттау
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Датасэтті жүктеу
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# (X) және мақсатты айнымалы (y)белгілерін анықтау
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']
# Деректерді оқыту және тест жиынтықтарына бөлу
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost үшін Matrix нысандарын құру
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Xgboost моделінің параметрлерін анықтау
params = {
    'objective': 'binary:logistic',  # Екілік жіктеу
    'max_depth': 3,                 # Ағаштардың тереңдігі
    'learning_rate': 0.1,           # Оқу жыламдығы
    'eval_metric': 'logloss'        # Өнімділікті бағалау көрсеткіші
}

# Xgboost моделін оқыту
num_round = 100
model = xgb.train(params, dtrain, num_round)

# Сынақ жиынтығында болжау
y_pred = model.predict(dtest)
y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]

# Модель өнімділігін бағалау
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Точность модели: {accuracy:.2f}')

