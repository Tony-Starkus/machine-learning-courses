import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

base_cartao = pd.read_csv('../Bases de dados/credit_card_clients.csv', header=1)

# Somando todos os valores para saber quanto cada cliente deve
base_cartao['BILL_TOTAL'] = base_cartao['BILL_AMT1'] + base_cartao['BILL_AMT2'] + base_cartao['BILL_AMT3'] + base_cartao['BILL_AMT4'] + base_cartao['BILL_AMT5'] + base_cartao['BILL_AMT6']

# Pegando o limite do cartão e a divida do cartão de cada cliente (colunas 1 e 25)
X_cartao = base_cartao.iloc[:, [1, 25]].values

scaler_cartao = StandardScaler()
X_cartao = scaler_cartao.fit_transform(X_cartao)

dbscan_cartao = DBSCAN(eps=0.37, min_samples=5)

rotulos = dbscan_cartao.fit_predict(X_cartao)
print(np.unique(rotulos, return_counts=True))

grafico = px.scatter(x=X_cartao[:, 0], y=X_cartao[:, 1], color=rotulos)
grafico.show()
