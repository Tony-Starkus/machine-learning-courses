import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from yellowbrick.classifier import ConfusionMatrix

with open("../Seção 3 - Pre-processamento/census.pkl", "rb") as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)

    print(f"{X_census_treinamento.shape=}")
    print(f"{y_census_treinamento.shape=}")
    print(f"{X_census_teste.shape=}")
    print(f"{y_census_teste.shape=}")

naive_census = GaussianNB()

# Criar tabela de probabilidades
naive_census.fit(X=X_census_treinamento, y=y_census_treinamento)

# Previsões sem saber a resposta certa
previsoes = naive_census.predict(X=X_census_teste)
print(f"{previsoes=}")

# Verificando o quanto as previsões acertaram comparando as respostas certas
print(f"Accuracy: {accuracy_score(y_true=y_census_teste, y_pred=previsoes)}")

cm = ConfusionMatrix(naive_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)
cm.show()

print(f"{classification_report(y_true=y_census_teste, y_pred=previsoes)}")
