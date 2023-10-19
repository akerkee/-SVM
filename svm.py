# Импортируем необходимые библиотеки
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Загрузим набор данных Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создадим модель SVM
svm_model = SVC(kernel='linear')  # Вы можете выбрать различные ядра: linear, rbf, poly, и т.д.

# Обучим модель на обучающей выборке
svm_model.fit(X_train, y_train)

# Сделаем прогнозы на тестовой выборке
y_pred = svm_model.predict(X_test)

# Оценим точность модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy}')'''
'''# Импортируем необходимые библиотеки
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Загрузим набор данных Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создадим модель T-SVM
class T_SVM(SVC):
    def fit(self, X, y, sample_weight=None):
        # Реализуйте здесь логику T-версии SVM
        # Это потребует создания кастомной функции потерь и настройки параметров SVM
        # Пример адаптации T-SVM можно найти в статьях и исследованиях

        # Временно используем обычную SVM
        super().fit(X, y, sample_weight)

# Создадим модель T-SVM
tsvm_model = T_SVM(kernel='linear')  # Вы можете выбрать различные ядра: linear, rbf, poly, и т.д.

# Обучим модель на обучающей выборке
tsvm_model.fit(X_train, y_train)

# Сделаем прогнозы на тестовой выборке
y_pred = tsvm_model.predict(X_test)

# Оценим точность модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy}')'''
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузим набор данных Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Создадим модель SVM-T вручную
class SVM_T:
    def _init_(self, C_t=1.0, kernel='linear'):
        self.C_t = C_t
        self.kernel = kernel

    def fit(self, X, y):
        # Реализуйте логику SVM-T здесь
        # Это потребует создания кастомной функции потерь и настройки параметров SVM-T
        # Примеры адаптации SVM-T можно найти в статьях и исследованиях

        # Временно используем обычную SVM
        from sklearn.svm import SVC
        self.svm_model = SVC(C=self.C_t, kernel=self.kernel)
        self.svm_model.fit(X, y)

    def predict(self, X):
        # Верните предсказания модели
        return self.svm_model.predict(X)

# Создадим модель SVM-T
svm_t_model = SVM_T(C_t=1.0, kernel='linear')
# Вы можете выбрать различные ядра: linear, rbf, poly, и т.д.

# Обучим модель на обучающей выборке
svm_t_model.fit(X_train, y_train)

# Сделаем прогнозы на тестовой выборке
y_pred = svm_t_model.predict(X_test)

# Оценим точность модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy}')
