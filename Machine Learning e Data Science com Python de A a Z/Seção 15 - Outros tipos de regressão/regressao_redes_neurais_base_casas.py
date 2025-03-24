import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

base_casas = pd.read_csv('../Bases de dados/house_prices.csv')
print(base_casas)

X_casas = base_casas.iloc[:, 0:1].values
y_casas = base_casas.iloc[:, 1].values

X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
    X_casas,
    y_casas,
    test_size=0.3,
    random_state=0
)

scaler_x = StandardScaler()
X_casas_treinamento_scaled = scaler_x.fit_transform(X_casas_treinamento)

scaler_y = StandardScaler()
y_casas_treinamento_scaled = scaler_y.fit_transform(y_casas_treinamento.reshape(-1,1))

regressor_rna_casas = MLPRegressor(max_iter=1000, hidden_layer_sizes=(9,9))
regressor_rna_casas.fit(X_casas_treinamento_scaled, y_casas_treinamento_scaled.ravel())
print(f"{regressor_rna_casas.score(X_casas_treinamento_scaled, y_casas_treinamento_scaled)=}")

grafico = px.scatter(x=X_casas_treinamento_scaled.ravel(), y=y_casas_treinamento_scaled.ravel())
grafico.add_scatter(x=X_casas_treinamento_scaled.ravel(), y=regressor_rna_casas.predict(X_casas_treinamento_scaled), name='Regressão')
grafico.show()

novo = [[40]]
novo = scaler_x.transform(novo)

print(f"{scaler_y.inverse_transform(regressor_rna_casas.predict(novo).reshape(-1,1))=}")
