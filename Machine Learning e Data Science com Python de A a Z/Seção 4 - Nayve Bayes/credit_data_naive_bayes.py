import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from yellowbrick.classifier import ConfusionMatrix

with open('../Seção 3 - Pre-processamento/credit.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

print(f"{X_credit_treinamento.shape=} {y_credit_treinamento.shape=}")
print(f"{X_credit_teste.shape=} {y_credit_teste.shape=}")

naive_credit_data = GaussianNB()
naive_credit_data.fit(X_credit_treinamento, y_credit_treinamento)

# Previsões sem saber a resposta certa
previsoes = naive_credit_data.predict(X_credit_teste)
print(f"{previsoes=}")

# Verificando o quanto as previsões acertaram comparando as respostas certas
print(f"{y_credit_teste=}")
print("Accuracy:", accuracy_score(y_true=y_credit_teste, y_pred=previsoes))
print(confusion_matrix(y_true=y_credit_teste, y_pred=previsoes))
"""
[[428   8]
 [ 23  41]]

Linha 0 => clientes que pagam emprestimo
    Linha 1 => Clientes que não pagam o emprestimo
    428 => Clientes que pagam e foram classificados corretamente como clientes que pagam
    8 => Clientes que pagam mas não foram classificados corretamente
    23 => Clientes que não pagam foram classificados como clients que pagam o emprestimo
    41 => Clientes que não pagam foram classificados corretamente como clientes que não pagam 
"""

cm = ConfusionMatrix(naive_credit_data)
cm.fit(X=X_credit_treinamento, y=y_credit_treinamento)
cm.score(X=X_credit_teste, y=y_credit_teste)
cm.show()

print(classification_report(y_true=y_credit_teste, y_pred=y_credit_teste))

