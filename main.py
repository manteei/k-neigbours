from random import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df = pd.read_csv('WineDataset.csv')

wine_column = df['Wine']

# Выделение числовых признаков для масштабирования
numerical_features = df.drop('Wine', axis=1)

# Масштабирование числовых признаков
numerical_features = (numerical_features - numerical_features.mean()) / numerical_features.std()
df_scaled = pd.concat([wine_column, numerical_features], axis=1)

# Обработка отсутствующих значений
df.fillna(df.mean(), inplace=True)

# Выделение числовых признаков и целевой переменной
X = df.drop('Wine', axis=1).values
y = df['Wine'].values

#Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def k_nearest_neighbors(X_train, y_train, x_test, k, selected_features):
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1[selected_features] - x2[selected_features]) ** 2))

    distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    result_class = np.bincount(k_nearest_labels).argmax()
    return result_class


# Создание и обучение Модели 1
random_features_model_1 = np.random.choice(X_train.shape[1], size=4, replace=False)
y_pred_model_1 = []
for x_test in X_test:
    result_class = k_nearest_neighbors(X_train, y_train, x_test, k=3, selected_features=random_features_model_1)
    y_pred_model_1.append(result_class)





# Создание и обучение Модели 2
y_pred_model_2 = []
fixed_features_model_2 = [0, 1, 9, 10, 11]
for x_test in X_test:
    result_class = k_nearest_neighbors(X_train, y_train, x_test, k=3, selected_features=fixed_features_model_2)
    y_pred_model_2.append(result_class)

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(model_name)
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, np.unique(y_true))
    plt.yticks(tick_marks, np.unique(y_true))
    plt.xlabel('Предсказание')
    plt.ylabel('Реальность')
    for i in range(len(np.unique(y_true))):
        for j in range(len(np.unique(y_true))):
            plt.text(j, i, str(cm[i, j]), horizontalalignment='center', verticalalignment='center', color='red',
                     fontsize=12)
    plt.show()


plot_confusion_matrix(y_test, y_pred_model_1, 'Модель 1')
plot_confusion_matrix(y_test, y_pred_model_2, 'Модель 2')

