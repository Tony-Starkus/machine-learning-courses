import pickle
import numpy as np
from matplotlib.scale import ScaleBase
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

rede_neural = pickle.load(open('rede_neural_finalizado.sav', 'rb'))
arvore = pickle.load(open('arvore_finalizado.sav', 'rb'))
svm = pickle.load(open('svm_finalizado.sav', 'rb'))

with open('../Bases de dados/credit.pkl', 'rb') as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

    X_credit = np.concatenate((X_credit_treinamento, X_credit_teste), axis=0)
    y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis=0)

    novo_registro = X_credit[1999]
    print(f"{novo_registro=}")
    print(f"{novo_registro.shape=}")
    novo_registro = novo_registro.reshape(1, -1)
    print(f"{novo_registro.shape=}")


    # COMBINAÇÃO DE CLASSIFICADORES
    resposta_rede_neural = rede_neural.predict(novo_registro)
    resposta_arvore = arvore.predict(novo_registro)
    resposta_svm = svm.predict(novo_registro)

    print(f"{resposta_rede_neural=}")
    print(f"{resposta_arvore=}")
    print(f"{resposta_svm=}")

    paga = 0
    nao_paga = 0

    if resposta_rede_neural[0] == 1:
        nao_paga += 1
    else:
        paga += 1

    if resposta_arvore[0] == 1:
        nao_paga += 1
    else:
        paga += 1

    if resposta_svm[0] == 1:
        nao_paga += 1
    else:
        paga += 1

    if paga > nao_paga:
        print('Cliente pagará o empréstimo')
    elif paga == nao_paga:
        print('Empate')
    else:
        print('Cliente não pagará o emprestimo')


    # REJEIÇÃO DE CLASSIFICADORES
    probabilidade_rede_neural = rede_neural.predict_proba(novo_registro)
    confianca_rede_neural = probabilidade_rede_neural.max()

    probabilidade_arvore = arvore.predict_proba(novo_registro)
    confianca_arvore = probabilidade_arvore.max()

    probabilidade_svm = svm.predict_proba(novo_registro)
    confianca_svm = probabilidade_svm.max()

    # 0.0 == 0% | 1.0 == 100%
    print(f"{probabilidade_rede_neural=}, {confianca_rede_neural=}")
    print(f"{probabilidade_arvore=}, {confianca_arvore=}")
    print(f"{probabilidade_svm=}, {confianca_svm=}")

    paga = 0
    nao_paga = 0
    confianca_minima = 0.999999
    algoritmos = 0

    if confianca_rede_neural >= confianca_minima:
        algoritmos += 1
        if resposta_rede_neural[0] == 1:
            nao_paga += 1
        else:
            paga += 1

    if confianca_arvore >= confianca_minima:
        algoritmos += 1
        if resposta_arvore[0] == 1:
            nao_paga += 1
        else:
            paga += 1

    if confianca_svm >= confianca_minima:
        algoritmos += 1

        if resposta_svm[0] == 1:
            nao_paga += 1
        else:
            paga += 1

    if paga > nao_paga:
        print(f'Cliente pagará o empréstimo, baseado em {algoritmos} algoritmos')
    elif paga == nao_paga:
        print(f'Empate, baseado em {algoritmos} algoritmos')
    else:
        print(f'Cliente não pagará o emprestimo, baseado em {algoritmos} algoritmos')

