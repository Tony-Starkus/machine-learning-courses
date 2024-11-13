import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

with open('../Bases de dados/credit.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)
    print(f"{X_credit_treinamento.shape=}")
    print(f"{y_credit_treinamento.shape=}")
    print(f"{X_credit_teste.shape=}")
    print(f"{y_credit_teste.shape=}")

    X_credit = np.concatenate((X_credit_treinamento, X_credit_teste), axis=0)
    y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)

    print(f"{X_credit.shape=}, {y_credit.shape=}")


    # create algorithms
    classificador_rede_neural = MLPClassifier(activation='relu', batch_size=64, solver='adam')
    classificador_rede_neural.fit(X_credit, y_credit)

    classificador_arvore = DecisionTreeClassifier(
        criterion='entropy',
        min_samples_leaf=1,
        min_samples_split=5,
        splitter='best'
    )
    classificador_arvore.fit(X_credit, y_credit)

    classificador_svm = SVC(C=2.0, kernel='rbf', probability=True)
    classificador_svm.fit(X_credit, y_credit)

    # save classifiers
    pickle.dump(classificador_rede_neural, open('rede_neural_finalizado.sav', 'wb'))
    pickle.dump(classificador_arvore, open('arvore_finalizado.sav', 'wb'))
    pickle.dump(classificador_svm, open('svm_finalizado.sav', 'wb'))
