import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
from matplotlib import pyplot as plt


with open('../Bases de dados/credit.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

    print(f"{X_credit_treinamento.shape=}, {y_credit_treinamento.shape=}")
    print(f"{X_credit_teste.shape=}, {y_credit_teste.shape=}")

knn_credit = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = knn_credit.predict(X_credit_teste)

accuracy = accuracy_score(y_credit_teste, previsoes)
print(f"{accuracy=}")

cm = ConfusionMatrix(knn_credit)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)
plt.show()

print(f"{classification_report(y_credit_teste, previsoes)}")
