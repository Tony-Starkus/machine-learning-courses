from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd


base_casas = pd.read_csv('../Bases de dados/house_prices.csv')

# Atributos previsores
X_casas = base_casas.iloc[:, 3:19].values

# Preços das casas
y_casas = base_casas.iloc[:, 2].values

X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
    X_casas,
    y_casas,
    test_size=0.3,
    random_state=0
)

print(f"{X_casas_treinamento.shape=}, {y_casas_treinamento.shape=}")
print(f"{X_casas_teste.shape=}, {y_casas_teste.shape=}")

regressor_multiplo_casas = LinearRegression()
regressor_multiplo_casas.fit(X_casas_treinamento, y_casas_treinamento)

print(f"B0 = {regressor_multiplo_casas.intercept_}")
print(f"{regressor_multiplo_casas.score(X_casas_treinamento, y_casas_treinamento)=}")
print(f"{regressor_multiplo_casas.score(X_casas_teste, y_casas_teste)=}")

previsoes = regressor_multiplo_casas.predict(X_casas_teste)
