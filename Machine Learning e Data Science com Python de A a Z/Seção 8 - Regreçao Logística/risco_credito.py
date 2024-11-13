from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import pickle
import numpy as np

with open('../Bases de dados/risco_credito.pkl', 'rb') as f:
    X_risco_credito, y_risco_credito = pickle.load(f)

    """
    A base de dados risco_credito tem 3 classificações: alto, moderado e baixo.
    Vamos remover os registros "moderado" e deixar apenas o "alto" e "baixo"
    """
    X_risco_credito = np.delete(X_risco_credito, [2, 7, 11], axis=0)
    y_risco_credito = np.delete(y_risco_credito, [2, 7, 11], axis=0)


logistic_risco_credito = LogisticRegression(random_state=1)
logistic_risco_credito.fit(X_risco_credito, y_risco_credito)

print(f"{logistic_risco_credito.intercept_=}")
print(f"{logistic_risco_credito.coef_=}")

previsoes1 = logistic_risco_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
print(f"{previsoes1=}")

plt.show()
