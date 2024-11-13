import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
from matplotlib import pyplot as plt


with open('../Bases de dados/census.pkl', 'rb') as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)

    print(f"{X_census_treinamento.shape=}, {y_census_treinamento.shape=}")
    print(f"{X_census_teste.shape=}, {y_census_teste.shape=}")


svm_census = SVC(kernel='linear', random_state=1)
svm_census.fit(X_census_treinamento, y_census_treinamento)

previsoes = svm_census.predict(X_census_teste)
accuracy = accuracy_score(y_census_teste, previsoes)

print(f"{accuracy}")

cm = ConfusionMatrix(svm_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)

print(classification_report(y_census_teste, previsoes))

plt.show()

