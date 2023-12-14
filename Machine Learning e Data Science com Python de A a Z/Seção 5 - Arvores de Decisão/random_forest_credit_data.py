from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
from matplotlib import pyplot as plt
import pickle

with open('./credit.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)
    print(f"{X_credit_treinamento.shape=}")
    print(f"{y_credit_treinamento.shape=}")
    print(f"{X_credit_teste.shape=}")
    print(f"{y_credit_teste.shape=}")

# random_state => gerar os mesmos resultados toda vez que o algoritmo rodar
# n_estimators => quantidade de arvores
random_forest_credit = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
random_forest_credit.fit(X=X_credit_treinamento, y=y_credit_treinamento)

previsoes = random_forest_credit.predict(X_credit_teste)
print(previsoes)
accuracy = accuracy_score(y_credit_teste, previsoes)
print(f"{accuracy=}")

cm = ConfusionMatrix(random_forest_credit)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)
plt.show()

print(f"{classification_report(y_credit_teste, previsoes)}")


