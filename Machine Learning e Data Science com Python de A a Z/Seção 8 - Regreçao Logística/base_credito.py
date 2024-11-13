from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
from matplotlib import pyplot as plt
import pickle

with open('../Bases de dados/credit.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)
    print(f"{X_credit_treinamento.shape=}")
    print(f"{y_credit_treinamento.shape=}")
    print(f"{X_credit_teste.shape=}")
    print(f"{y_credit_teste.shape=}")

logistic_credit = LogisticRegression(random_state=1)
logistic_credit.fit(X_credit_treinamento, y_credit_treinamento)
print(f"{logistic_credit.intercept_}")
print(f"{logistic_credit.coef_}")

previsoes = logistic_credit.predict(X_credit_teste)

acurracy = accuracy_score(y_credit_teste, previsoes)
cm = ConfusionMatrix(logistic_credit)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)
plt.show()
print(f"{acurracy=}")

print(f"{classification_report(y_credit_teste, previsoes)}")
