import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


base_casas = pd.read_csv('../Bases de dados/house_prices.csv')
print(base_casas)

X_casas = base_casas.iloc[:,5:6].values
y_casas = base_casas.iloc[:,2].values

print(f"idades: {X_casas}")
print(f"custos: {y_casas}")

X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(
    X_casas,
    y_casas,
    test_size=0.3,
    random_state=0
)

scaler_x_casas = StandardScaler()
X_casas_treinamento_scaled = scaler_x_casas.fit_transform(X_casas_treinamento)

scaler_y_casas = StandardScaler()
y_casas_treinamento_scaled = scaler_y_casas.fit_transform(y_casas_treinamento.reshape(-1,1))

print(f"{X_casas_treinamento_scaled=}")
print(f"{y_casas_treinamento_scaled=}")

X_casas_teste_scaled = scaler_x_casas.transform(X_casas_teste)
y_casas_teste_scaled = scaler_y_casas.transform(y_casas_teste.reshape(-1,1))
print(f"{X_casas_teste_scaled=}")
print(f"{y_casas_teste_scaled=}")

regressor_svr_casas = SVR(kernel='rbf')
regressor_svr_casas.fit(X_casas_treinamento_scaled, y_casas_treinamento_scaled.ravel())

print(f"{regressor_svr_casas.score(X_casas_treinamento_scaled, y_casas_treinamento_scaled)=}")
print(f"{regressor_svr_casas.score(X_casas_teste_scaled, y_casas_teste_scaled)=}")

previsoes = regressor_svr_casas.predict(X_casas_teste_scaled)

y_casas_teste_inverse = scaler_y_casas.inverse_transform(y_casas_teste_scaled)
previsoes_inverse = scaler_y_casas.inverse_transform(previsoes)

print(f"{mean_absolute_error(y_casas_teste_inverse, previsoes_inverse)}")
