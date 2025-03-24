import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

base_plano_saude2 = pd.read_csv('../Bases de dados/plano_saude2.csv')
print(base_plano_saude2)

X_plano_saude2 = base_plano_saude2.iloc[:, 0:1].values
y_plano_saude2 = base_plano_saude2.iloc[:, 1].values

print(f"idades: {X_plano_saude2}")
print(f"custos: {y_plano_saude2}")

# degree=2 => vai elevar cada valor da dos custos ao quadrado.
poly = PolynomialFeatures(degree=2)
X_plano_saude2_poly = poly.fit_transform(X_plano_saude2)
print(f"{X_plano_saude2_poly=}")

regressor_saude_polinomial = LinearRegression()
regressor_saude_polinomial.fit(X_plano_saude2_poly, y_plano_saude2)

# b0
print(f"b0: {regressor_saude_polinomial.intercept_}")

# b1
print(f"b0: {regressor_saude_polinomial.coef_}")

novo = [[40]]
novo_poly = poly.transform(novo)
print(f"{regressor_saude_polinomial.predict(novo_poly)=}")

previsoes = regressor_saude_polinomial.predict(X_plano_saude2_poly)

grafico = px.scatter(x=X_plano_saude2.ravel(), y=y_plano_saude2)
grafico.add_scatter(x=X_plano_saude2.ravel(), y=previsoes, name='Regressão')
grafico.show()
