import plotly.express as px
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures

base_plano_saude2 = pd.read_csv('../Bases de dados/plano_saude2.csv')
print(base_plano_saude2)

X_plano_saude2 = base_plano_saude2.iloc[:, 0:1].values
y_plano_saude2 = base_plano_saude2.iloc[:, 1].values

print(f"idades: {X_plano_saude2}")
print(f"custos: {y_plano_saude2}")

regressor_svr_saude_kernel_linear = SVR(kernel='linear')
regressor_svr_saude_kernel_linear.fit(X_plano_saude2, y_plano_saude2)
print(f"{regressor_svr_saude_kernel_linear=}")

grafico = px.scatter(x=X_plano_saude2.ravel(), y=y_plano_saude2)
grafico.add_scatter(x=X_plano_saude2.ravel(), y=regressor_svr_saude_kernel_linear.predict(X_plano_saude2), name='Regressão')
grafico.show()


regressor_svr_saude_kernel_poly = SVR(kernel='poly', degree=3)
regressor_svr_saude_kernel_poly.fit(X_plano_saude2, y_plano_saude2)
print(f"{regressor_svr_saude_kernel_poly=}")

grafico = px.scatter(x=X_plano_saude2.ravel(), y=y_plano_saude2)
grafico.add_scatter(x=X_plano_saude2.ravel(), y=regressor_svr_saude_kernel_poly.predict(X_plano_saude2), name='Regressão')
grafico.show()



regressor_svr_saude_kernel_rbf = SVR(kernel='rbf')
print(f"{regressor_svr_saude_kernel_rbf=}")

scaler_x = StandardScaler()
X_plano_saude2_scaled = scaler_x.fit_transform(X_plano_saude2)

scaler_y = StandardScaler()
y_plano_saude2_scaled = scaler_y.fit_transform(y_plano_saude2.reshape(-1,1))

regressor_svr_saude_kernel_rbf.fit(X_plano_saude2_scaled, y_plano_saude2_scaled.ravel())

grafico = px.scatter(x=X_plano_saude2_scaled.ravel(), y=y_plano_saude2_scaled.ravel())
grafico.add_scatter(x=X_plano_saude2_scaled.ravel(), y=regressor_svr_saude_kernel_rbf.predict(X_plano_saude2_scaled), name='Regressão')
grafico.show()

novo = scaler_x.transform([[40]])
print(f"{scaler_y.inverse_transform(regressor_svr_saude_kernel_rbf.predict(novo).reshape(-1,1))=}")
