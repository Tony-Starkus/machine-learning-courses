import numpy as np
from numpy.ma.core import shape
import seaborn as sns
from statsmodels.stats.multicomp import MultiComparison
from scipy.stats import f_oneway
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold

from scipy.stats import shapiro

import pandas as pd
import pickle

# Tuning dos parâmetros com GridSearch
"""
Tuning de hiperparâmetros, também conhecido como otimização de hiperparâmetros, refere-se ao processo
de escolher um conjunto ideal de hiperparâmetros para um determinado algoritmo de aprendizado de máquina.
"""

with open('../Bases de dados/credit.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)
    print(f"{X_credit_treinamento.shape=}")
    print(f"{y_credit_treinamento.shape=}")
    print(f"{X_credit_teste.shape=}")
    print(f"{y_credit_teste.shape=}")


X_credit = np.concatenate((X_credit_treinamento, X_credit_teste), axis=0)
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)
print(f"{X_credit.shape}")
print(f"{y_credit.shape}")

parametros = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}

grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=parametros)
grid_search.fit(X_credit, y_credit)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(f"{melhores_parametros=} {melhor_resultado=}")

# Random Forest
parametros = {
    'criterion': ['gini', 'entropy'],
    'n_estimators': [10, 40, 100, 150],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parametros)
grid_search.fit(X_credit, y_credit)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(f"RandomForest: {melhores_parametros=} {melhor_resultado=}")


# Knn
knn_parametros = {
    'n_neighbors': [3, 5, 10, 20],
    'p': [1, 2, 100, 150],
}
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=knn_parametros)
grid_search.fit(X_credit, y_credit)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(f"Knn: {melhores_parametros=} {melhor_resultado=}")


# Logistic Regression
logistic_regression_parametros = {
    'tol': [0.0001, 0.00001, 0.000001],
    'C': [1.0, 1.5, 2.0],
    'solver': ['lbfgs', 'sag', 'saga'],
}
grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=logistic_regression_parametros)
grid_search.fit(X_credit, y_credit)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(f"Logistic Regression: {melhores_parametros=} {melhor_resultado=}")


# SVM
svm_parametros = {
    'tol': [0.001, 0.0001],
    'C': [1.0, 1.5, 2.0],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
}
grid_search = GridSearchCV(estimator=SVC(), param_grid=svm_parametros)
grid_search.fit(X_credit, y_credit)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(f"SVM: {melhores_parametros=} {melhor_resultado=}")


# Redes Neurais
redes_neurais_parametros = {
    'activation': ['relu', 'logistic', 'tanh'],
    'solver': ['adam', 'sgd'],
    'batch_size': [10, 56],
}
grid_search = GridSearchCV(estimator=MLPClassifier(), param_grid=redes_neurais_parametros)
grid_search.fit(X_credit, y_credit)
melhores_parametros = grid_search.best_params_
melhor_resultado = grid_search.best_score_
print(f"Redes Neurais: {melhores_parametros=} {melhor_resultado=}")


resultados_arvore = []
resultados_random_forest = []
resultados_knn = []
resultados_logistica = []
resultados_svm = []
resultados_rede_neural = []

# 30 testees
for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    arvore = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, splitter='best')
    scores = cross_val_score(arvore, X_credit, y_credit, cv=kfold)
    resultados_arvore.append(scores.mean())

    random_forest = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=10)
    scores = cross_val_score(random_forest, X_credit, y_credit, cv=kfold)
    resultados_random_forest.append(scores.mean())

    knn = KNeighborsClassifier()
    scores = cross_val_score(knn, X_credit, y_credit, cv=kfold)
    resultados_knn.append(scores.mean())

    logistica = LogisticRegression(C=1.0, solver='lbfgs', tol=0.0001)
    scores = cross_val_score(logistica, X_credit, y_credit, cv=kfold)
    resultados_logistica.append(scores.mean())

    svm = SVC(kernel='rbf', C=2.0)
    scores = cross_val_score(svm, X_credit, y_credit, cv=kfold)
    resultados_svm.append(scores.mean())

    rede_neural = MLPClassifier(activation='relu', batch_size=56, solver='adam')
    scores = cross_val_score(rede_neural, X_credit, y_credit, cv=kfold)
    resultados_rede_neural.append(scores.mean())

print(f"{resultados_rede_neural=}")

resultados = pd.DataFrame({
    'Arvore': resultados_arvore,
    'Random forest': resultados_random_forest,
    'KNN': resultados_knn,
    'Logistica': resultados_logistica,
    'SVM': resultados_svm,
    'Rede Neural': resultados_rede_neural,
})

print(resultados)
print(resultados.describe())

# APLICAR TESTE DE SHAPIRO

# Definindo a confiança do teste em 95%
alpha = 0.05
shapiro(resultados_arvore)
shapiro(resultados_random_forest)
shapiro(resultados_knn)
shapiro(resultados_logistica)
shapiro(resultados_svm)
shapiro(resultados_rede_neural)

sns.displot(resultados_arvore, kind='kde')
sns.displot(resultados_random_forest, kind='kde')
sns.displot(resultados_knn, kind='kde')
sns.displot(resultados_logistica, kind='kde')
sns.displot(resultados_svm, kind='kde')
sns.displot(resultados_rede_neural, kind='kde')

# TESTE DE HIPÓTESE COM ANOVA E TUKEY
p = f_oneway(resultados_arvore, resultados_random_forest, resultados_knn, resultados_logistica, resultados_svm, resultados_rede_neural)
print(f"{p=}")

alpha = 0.05

if p <= alpha:
    print("Hipótese nula rejeitada. Daddos são diferentes")
else:
    print("Hipótese alternativa rejeitada. Resultados são iguais")


resultados_algoritmos = {'accuracy': np.concatenate([resultados_arvore, resultados_random_forest, resultados_knn, resultados_logistica, resultados_svm, resultados_rede_neural]),
                         'algoritmo': ['arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore','arvore',
                          'random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest','random_forest',
                          'knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn','knn',
                          'logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica','logistica',
                          'svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm','svm',
                          'rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural','rede_neural']}


resultados_df = pd.DataFrame(resultados_algoritmos)
print(f"{resultados_df=}")

compara_algoritmos = MultiComparison(resultados_df['accuracy'], resultados_df['algoritmo'])
teste_estatistico = compara_algoritmos.tukeyhsd()
print("teste_statistico")
print(teste_estatistico)

teste_estatistico.plot_simultaneous()
