import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd


# PREVER O PREÇO DA CASA COM BASE NO METRO QUADRADO
base_casas = pd.read_csv('../Bases de dados/house_prices.csv')
print(base_casas.describe())

# Verificando se tem alguma linha faltando dado
print(base_casas.isnull().sum())

# Cálculo de correlação
# print(base_casas.corr())

figura = plt.figure(figsize=(20,20))
sns.heatmap(base_casas.corr())

# Pegando as linhas da coluna de metragem
X_casas = base_casas.iloc[:,5:6].values
print(X_casas)

# Pegando as linhas da coluna de preço das casas
y_casas = base_casas.iloc[:,2].values
print(f"{y_casas=}")

# 0.3 => 30%
X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
    X_casas,
    y_casas,
    test_size=0.3,
    random_state=0
)

print(f"{X_casas_treinamento.shape=}, {y_casas_treinamento.shape=}")
print(f"{X_casas_teste.shape=}, {y_casas_teste.shape=}")

regressor_simples_casas = LinearRegression()
regressor_simples_casas.fit(X_casas_treinamento, y_casas_treinamento)
print(f"B0 => {regressor_simples_casas.intercept_}")
print(f"B1 => {regressor_simples_casas.coef_}")
print(f"R2 (treinamento) => {regressor_simples_casas.score(X_casas_treinamento, y_casas_treinamento)}")
print(f"R2 (teste) => {regressor_simples_casas.score(X_casas_teste, y_casas_teste)}")

previsoes = regressor_simples_casas.predict(X_casas_treinamento)

grafico = px.scatter(x=X_casas_treinamento.ravel(), y=previsoes)

grafico1 = px.scatter(x=X_casas_treinamento.ravel(), y=y_casas_treinamento)
grafico2 = px.line(x=X_casas_treinamento.ravel(), y=previsoes)
grafico2.data[0].line.color = 'red'
grafico3 = go.Figure(data=grafico1.data + grafico2.data)

grafico3.show()

previsoes_teste = regressor_simples_casas.predict(X_casas_teste)
print(f"Mean Absolute Error (MAE)={abs(y_casas_teste - previsoes_teste).mean()=}")
print(f"{mean_absolute_error(y_casas_teste, previsoes_teste)}")
print(f"{mean_squared_error(y_casas_teste, previsoes_teste)}")

grafico1 = px.scatter(x=X_casas_teste.ravel(), y=y_casas_teste)
grafico2 = px.line(x=X_casas_teste.ravel(), y=previsoes_teste)
grafico2.data[0].line.color = 'red'
grafico3 = go.Figure(data=grafico1.data + grafico2.data)
