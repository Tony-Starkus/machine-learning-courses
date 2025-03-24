import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

base_casas = pd.read_csv('../Bases de dados/house_prices.csv')
print(base_casas.describe())

# Verificando se tem alguma linha faltando dado
print(base_casas.isnull().sum())

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

poly = PolynomialFeatures(degree=2)

X_casas_treinamento_poly = poly.fit_transform(X_casas_treinamento)
X_casas_teste_poly = poly.transform(X_casas_teste)

print(f"{X_casas_treinamento_poly.shape=} {X_casas_teste_poly.shape=}")

regressor_casas_poly = LinearRegression()
regressor_casas_poly.fit(X_casas_treinamento_poly, y_casas_treinamento)

print(f"{regressor_casas_poly.score(X_casas_treinamento_poly, y_casas_treinamento)=}")
print(f"{regressor_casas_poly.score(X_casas_teste_poly, y_casas_teste)=}")

previsoes = regressor_casas_poly.predict(X_casas_teste_poly)

print(f"{mean_absolute_error(y_casas_teste, previsoes)=}")
