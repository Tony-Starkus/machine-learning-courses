import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor


base_plano_saude2 = pd.read_csv('../Bases de dados/plano_saude2.csv')
print(base_plano_saude2)

X_plano_saude2 = base_plano_saude2.iloc[:, 0:1].values
y_plano_saude2 = base_plano_saude2.iloc[:, 1].values

# n_estimators => número de arvores
regressor_random_forest_saude = RandomForestRegressor(n_estimators=10)
regressor_random_forest_saude.fit(X_plano_saude2, y_plano_saude2)

print(f"{regressor_random_forest_saude.score(X_plano_saude2, y_plano_saude2)=}")

# Pegando idade mínima e máxima e vamos incrementando + 0.1
X_teste_saude = X_plano_saude2.reshape(-1,1)
print(f"{X_teste_saude}")
print(f"{X_teste_saude.shape}")

grafico = px.scatter(x=X_plano_saude2.ravel(), y=y_plano_saude2)
grafico.add_scatter(x=X_teste_saude.ravel(), y=regressor_random_forest_saude.predict(X_teste_saude), name='Regressão')
grafico.show()

print(f"{regressor_random_forest_saude.predict([[40]])=}")
