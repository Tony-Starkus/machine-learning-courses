import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

"""
We need to determine the default collum to know if client may pay the loan
CREDIT DATA COLUMNS DESCRIPTION
client_id -> the id of client
income -> 
age -> the client age
loan ->
default -> Define if client paid the loan. Zero says that loan is paid, one says client doesn't paid loan
"""

# 👇️ set tkinter backend
matplotlib.use('TkAgg')

# Read csv data file
base_credit = pd.read_csv('credit_data.csv')

# print(base_credit)
# Print some statistics
# print(base_credit.describe())
# print(base_credit[base_credit['income'] >= 69995.685578])
# print(base_credit[base_credit['loan'] <= 1.377630])

# CLEAR INCONSISTENT DATA FROM DATABASE
# Delete invalid ages data
# base_credit_normalized = base_credit.drop(base_credit[base_credit['age'] < 0].index)
# print(f"{base_credit_normalized=}")


# Fix inconsistent age data using the age mean
# print(base_credit.loc[base_credit['age'] < 0])
mean_age = base_credit['age'][base_credit['age'] > 0].mean()
base_credit.loc[base_credit['age'] < 0, 'age'] = mean_age
# print(f"{base_credit.head(20)}")
# print(base_credit.loc[base_credit['age'] < 0])


# print(np.unique(base_credit['default'], return_counts=True))
sns.countplot(x=base_credit['default'])
plt.hist(x=base_credit['loan'])
# plt.show()

# Show matriz to know the range of datas
# grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')
# grafico.show()


# AULA 19 - FIX MISSING DATA
# We can 3 row that doesn't have age (NaN)
print(base_credit.isnull().sum())
print(base_credit.loc[pd.isnull(base_credit['age'])])
# Set NaN age with mean of age column
base_credit['age'].fillna(base_credit['age'].mean(), inplace=True)
# print(base_credit.isnull().sum())

# AULA 20
X_credit = base_credit.iloc[:, 1:4].values
print(f"{X_credit=}")

y_credit = base_credit.iloc[:, 4].values
print(f"{y_credit=}")

# AULA 21
# 0 => Income | 1 => Age | 2=> Loan
print(X_credit[:, 0].min(), X_credit[:, 1].min(), X_credit[:, 2].min())
print(X_credit[:, 0].max(), X_credit[:, 1].max(), X_credit[:, 2].max())

"""
Com base nas prints anteriores, vemos que a distância dos números é muito grande,
e isso resulta em cálculos mais pesados.
Para diminuir o esforço, podemos usar dois algoritmos:

Padronização (Standardisation) => É usado quando tem muito outliers (valores que estão muito fora do padrão)
Normalização (Normalization) => 
"""
scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit)
print(X_credit[:, 0].min(), X_credit[:, 1].min(), X_credit[:, 2].min())
print(X_credit[:, 0].max(), X_credit[:, 1].max(), X_credit[:, 2].max())

print("=============== DIVISÃO DAS BASES EM TREINAMENTO E TESTE ===============")
X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(
    X_credit,
    y_credit,
    test_size=0.25,
    random_state=0  # When you run code again, the "0" means to have the same split result between run code
)

print(f"{X_credit_treinamento.shape=}")
print(f"{y_credit_treinamento.shape=}")
print(f"{X_credit_teste.shape=}, {y_credit_teste.shape=}")

with open('credit.pkl', mode='wb') as f:
    pickle.dump([X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste], f)
