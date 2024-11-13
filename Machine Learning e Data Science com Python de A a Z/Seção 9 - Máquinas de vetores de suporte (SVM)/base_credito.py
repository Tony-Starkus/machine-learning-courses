from sklearn.svm import SVC
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

# C -> Quanto mais o valor, maior a tendência do algoritmo encontrar as linhas entre as classes
svm_credit = SVC(kernel='rbf', random_state=1, C=2.0)

svm_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = svm_credit.predict(X_credit_teste)
accuracy = accuracy_score(y_credit_teste, previsoes)
print(f"{accuracy=}")

cm = ConfusionMatrix(svm_credit)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)

print(classification_report(y_credit_teste, previsoes))

plt.show()
