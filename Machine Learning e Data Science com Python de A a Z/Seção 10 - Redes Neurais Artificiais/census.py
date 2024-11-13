import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier  # Multi Layer Perceptron
from yellowbrick.classifier import ConfusionMatrix
from matplotlib import pyplot as plt

with open('../Bases de dados/census.pkl', 'rb') as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)

    print(f"{X_census_treinamento.shape=}, {y_census_treinamento.shape=}")
    print(f"{X_census_teste.shape=}, {y_census_teste.shape=}")

#
# 55

"""
Esta rede neural está configurada da seguinte forma:
 - 108 Neurônios na camada de entrada
 - 55 neurônios na primeira cada oculta
 - 55 neurônios na segunda camada oculta
 - 1 neurônio na camada de saída
"""
rede_neural_census = MLPClassifier(
    verbose=True,
    max_iter=1000,
    tol=0.000010,
    hidden_layer_sizes=(55, 55)
)

rede_neural_census.fit(X_census_treinamento, y_census_treinamento)

previsoes = rede_neural_census.predict(X_census_teste)
acurracy = accuracy_score(y_census_teste, previsoes)
print(f"{acurracy=}")

cm = ConfusionMatrix(rede_neural_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)

print(classification_report(y_census_teste, previsoes))

plt.show()
