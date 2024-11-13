import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.constants import grain
from yellowbrick.regressor import ResidualsPlot

from sklearn.linear_model import LinearRegression

base_plano_saude = pd.read_csv('../Bases de dados/plano_saude.csv')

print("BASE DE DADOS - PLANO DE SAÚDE")
print(f"{base_plano_saude}")

# Pegar as idades
X_plano_saude = base_plano_saude.iloc[:, 0].values
print(f"{X_plano_saude=}, {X_plano_saude.shape=}")

# Pega os custos
y_plano_saude = base_plano_saude.iloc[:, 1].values
print(f"{y_plano_saude}")

print(f"{np.corrcoef(X_plano_saude, y_plano_saude)}")

# Transformando o array em uma matriz (N linhas | 1 Coluna
X_plano_saude = X_plano_saude.reshape(-1,1)
print(f"{X_plano_saude}")
print(f"{X_plano_saude.shape}")

# Treinando modelo
regressor_plano_saude = LinearRegression()
regressor_plano_saude.fit(X_plano_saude, y_plano_saude)


# b0 e b1 definem a localização da linha (treinamento) no plano cartesiano (gráfico)
# b0
print(f"{regressor_plano_saude.intercept_=}")
# b1
print(f"{regressor_plano_saude.coef_=}")

previsoes = regressor_plano_saude.predict(X_plano_saude)

# Grafico de dispersão

grafico = px.scatter(x=X_plano_saude.ravel(), y=y_plano_saude)
grafico.add_scatter(x=X_plano_saude.ravel(), y=previsoes, name='Regressão')
grafico.show()

# Desempenho do modelo
print(f"Desempenho do modelo: {regressor_plano_saude.score(X_plano_saude, y_plano_saude)}")

visualizador = ResidualsPlot(regressor_plano_saude)
visualizador.fit(X_plano_saude, y_plano_saude)
visualizador.poof()
