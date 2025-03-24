import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

base_cartao = pd.read_csv('../Bases de dados/credit_card_clients.csv', header=1)

# Somando todos os valores para saber quanto cada cliente deve
base_cartao['BILL_TOTAL'] = base_cartao['BILL_AMT1'] + base_cartao['BILL_AMT2'] + base_cartao['BILL_AMT3'] + base_cartao['BILL_AMT4'] + base_cartao['BILL_AMT5'] + base_cartao['BILL_AMT6']
print(base_cartao)

# Pegando o limite do cartão e a divida do cartão de cada cliente (colunas 1 e 25)
X_cartao = base_cartao.iloc[:, [1, 25]].values

scaler_cartao = StandardScaler()
X_cartao = scaler_cartao.fit_transform(X_cartao)

wcss = []
# Verificando quantos clustes é o ideal para utilizar.
for i in range(1, 11):
     kmeans_cartao = KMeans(n_clusters=i, random_state=0)
     kmeans_cartao.fit(X_cartao)
     wcss.append(kmeans_cartao.inertia_)

print(wcss)

grafico = px.line(x=range(1, 11), y=wcss)
grafico.show()

# Selecionando o número de clustes em que a queda de Y em relação a X é menor
kmeans_cartao = KMeans(n_clusters=4, random_state=0)
rotulos = kmeans_cartao.fit_predict(X_cartao)

grafico = px.scatter(x=X_cartao[:, 0], y=X_cartao[:, 1], color=rotulos)
grafico.show()

lista_clientes = np.column_stack((base_cartao, rotulos))
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]
print(lista_clientes)
