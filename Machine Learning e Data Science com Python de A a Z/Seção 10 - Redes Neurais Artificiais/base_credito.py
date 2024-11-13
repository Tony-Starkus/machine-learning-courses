from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier  # Multi Layer Perceptron
from yellowbrick.classifier import ConfusionMatrix
from matplotlib import pyplot as plt
import pickle

with open('../Bases de dados/credit.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)
    print(f"{X_credit_treinamento.shape=}")
    print(f"{y_credit_treinamento.shape=}")
    print(f"{X_credit_teste.shape=}")
    print(f"{y_credit_teste.shape=}")

rede_neural_credit = MLPClassifier(
    max_iter=1500,
    verbose=True,
    tol=0.0000100,
    solver='adam',
    activation='relu',
    hidden_layer_sizes=(50, 50)
)

rede_neural_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = rede_neural_credit.predict(X_credit_teste)
acurracy = accuracy_score(y_credit_teste, previsoes)
print(f"{acurracy=}")

cm = ConfusionMatrix(rede_neural_credit)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)

print(classification_report(y_credit_teste, previsoes))

plt.show()
