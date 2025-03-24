import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor


base_plano_saude2 = pd.read_csv('../Bases de dados/plano_saude2.csv')
print(base_plano_saude2)

X_plano_saude2 = base_plano_saude2.iloc[:, 0:1].values
y_plano_saude2 = base_plano_saude2.iloc[:, 1].values

regressor_arvore_saude = DecisionTreeRegressor()
regressor_arvore_saude.fit(X_plano_saude2, y_plano_saude2)

previsoes = regressor_arvore_saude.predict(X_plano_saude2)

# Resultado = 1 => significa que os resultados das previsões são todos iguais
print(f"{regressor_arvore_saude.score(X_plano_saude2, y_plano_saude2)=}")

grafico = px.scatter(x=X_plano_saude2.ravel(), y=y_plano_saude2)
grafico.add_scatter(x=X_plano_saude2.ravel(), y=previsoes, name='Regressão')
grafico.show()

# Pegando idade mínimae máxima e vamos incrementando + 0.1
X_teste_arvore = np.arange(min(X_plano_saude2), max(X_plano_saude2), step=0.1)
print(f"{X_teste_arvore=}")
print(f"{X_teste_arvore.shape=}")

X_teste_arvore = X_teste_arvore.reshape(-1, 1)
print(f"{X_teste_arvore.shape=}")

grafico = px.scatter(x=X_plano_saude2.ravel(), y=y_plano_saude2)
grafico.add_scatter(x=X_teste_arvore.ravel(), y=regressor_arvore_saude.predict(X_teste_arvore), name='Regressão')
grafico.show()

print(f"{regressor_arvore_saude.predict([[40]])=}")
