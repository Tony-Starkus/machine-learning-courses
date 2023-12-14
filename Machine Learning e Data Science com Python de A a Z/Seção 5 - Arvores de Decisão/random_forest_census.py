from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
from matplotlib import pyplot as plt
import pickle

with open('./census.pkl', 'rb') as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)
    print(f"{X_census_treinamento.shape=}")
    print(f"{y_census_treinamento.shape=}")
    print(f"{X_census_teste.shape=}")
    print(f"{y_census_teste.shape=}")

# random_state => gerar os mesmos resultados toda vez que o algoritmo rodar
# n_estimators => quantidade de arvores
random_forest_credit = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
random_forest_credit.fit(X=X_census_treinamento, y=y_census_treinamento)

previsoes = random_forest_credit.predict(X_census_teste)
print(previsoes)
accuracy = accuracy_score(y_census_teste, previsoes)
print(f"{accuracy=}")

cm = ConfusionMatrix(random_forest_credit)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)
plt.show()

print(f"{classification_report(y_census_teste, previsoes)}")


