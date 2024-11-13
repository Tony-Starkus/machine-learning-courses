import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
from matplotlib import pyplot as plt


with open('../Bases de dados/census.pkl', 'rb') as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)

    print(f"{X_census_treinamento.shape=}, {y_census_treinamento.shape=}")
    print(f"{X_census_teste.shape=}, {y_census_teste.shape=}")

knn_census = KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=2)
knn_census.fit(X_census_treinamento, y_census_treinamento)

previsoes = knn_census.predict(X_census_teste)

accuracy = accuracy_score(y_census_teste, previsoes)
print(f"{accuracy=}")

cm = ConfusionMatrix(knn_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)
plt.show()

print(f"{classification_report(y_census_teste, previsoes)}")
