import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
from matplotlib import pyplot as plt

with open('./census.pkl', 'rb') as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)

    print(X_census_treinamento.shape, y_census_treinamento.shape)
    print(X_census_teste.shape, y_census_teste.shape)

arvore_census = DecisionTreeClassifier(criterion='entropy', random_state=0)
arvore_census.fit(X=X_census_treinamento, y=y_census_treinamento)

previsoes = arvore_census.predict(X_census_teste)
print(f"{accuracy_score(y_census_teste, previsoes)=}")

cm = ConfusionMatrix(arvore_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)
plt.show()

print(classification_report(y_census_teste, previsoes))
