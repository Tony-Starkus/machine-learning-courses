import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle

base_census = pd.read_csv('census.csv')
# Print some statistics
print(base_census.describe())

# Check if exists any row with invalid value
print(base_census.isnull().sum())


print("=============== DATA ANALITICS ===============")
print(np.unique(base_census['income'], return_counts=True))

sns.countplot(x=base_census['income'])
plt.title("INCOME")
plt.show()

plt.hist(x=base_census['age'])
plt.title("AGE")
plt.show()

plt.hist(x=base_census['education-num'])
plt.title('EDUCATION TIME')
plt.show()

plt.hist(x=base_census['hour-per-week'])
plt.title('WORK HOURS PER WEEK')
plt.show()

grafico = px.treemap(base_census, path=['workclass', 'age'])
grafico.show()

grafico_2 = px.treemap(base_census, path=['occupation', 'relationship', 'age'])
grafico_2.show()

grafico_categorias_paralelas = px.parallel_categories(base_census, dimensions=['occupation', 'relationship'])
grafico_categorias_paralelas.show()

grafico_categorias_paralelas_2 = px.parallel_categories(
    base_census,
    dimensions=['workclass', 'occupation', 'relationship']
)
grafico_categorias_paralelas_2.show()

print("=============== DIVISÃO DE PREVISORES E CLASSES ===============")

# Previsores (Selecionamos todas as colunas com excessão do "income", pois é ele que queremos fazer a previsão)
X_census = base_census.iloc[:, 0:14].values
print(base_census.columns)

# Classes
y_census = base_census.iloc[:, 14].values
print(f"{X_census=}")
print(f"{y_census=}")

print("=============== TRATAMENTO DE ATRIBUTOS CATEGÓRICOS - LabelEncoder ===============")

# LabelEncoder
# No machine learning, são feitos muitos cálculos, e valores do tipo nominal não pode ser calculados
# Nós transformamos dados string em números
label_encoder_teste = LabelEncoder()

# Estamos pegando todas as linhas da coluna "Workclass"
teste = label_encoder_teste.fit_transform(X_census[:, 1])
print(f"teste_label_encoder: {teste}")

# Para cada atributo nominal, nós criamos uma variável de label encoder
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()
print(f"{X_census[0]=}")
print(f"{X_census[1]=}")

# Agora acessamos todos as linhas de cada coluna para alteramos os valores nominais por representações numéricas
X_census[:, 1] = label_encoder_workclass.fit_transform(X_census[:, 1])
X_census[:, 3] = label_encoder_education.fit_transform(X_census[:, 3])
X_census[:, 5] = label_encoder_marital.fit_transform(X_census[:, 5])
X_census[:, 6] = label_encoder_occupation.fit_transform(X_census[:, 6])
X_census[:, 7] = label_encoder_relationship.fit_transform(X_census[:, 7])
X_census[:, 8] = label_encoder_race.fit_transform(X_census[:, 8])
X_census[:, 9] = label_encoder_sex.fit_transform(X_census[:, 9])
X_census[:, 13] = label_encoder_country.fit_transform(X_census[:, 13])
print(f"{X_census[0]=}")
print(f"{X_census[1]=}")

print("=============== TRATAMENTO DE ATRIBUTOS CATEGÓRICOS - OneHotEncoder ===============")
"""
Um problema do LabelEncoder, ao substituir atributos nominais por representação numérica, é que em
algoritmos que trabalham com pesos, o atributo pode erroneamente dar mais valor para os dados.
Por exemplo, se temos três carros: Gol, Pálio e Uno; e o LabelEncoder atribui 1, 2 e 3 respectivamente, o Uno
vai ser considrado mais importante nos cálculos, sendo que na realidade o número 3 é apenas uma representação
da palavra Uno e não deve ser considerado nos cálculos.

Com o OneHotEncoder, ele vai criar um mapa dos dados e vai atribui nesse mapa 0 e 1 para cada linha de acordo
com o valor daquele dado.
Gol   1 0 0
Pálio 0 1 0
Uno   0 0 1 # Encode
"""

onehotencoder_census = ColumnTransformer(
    transformers=[('OneHot', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],
    remainder='passthrough'
)

X_census = onehotencoder_census.fit_transform(X_census).toarray()
print(f"{X_census[0]=}")

print("=============== ESCALONAMENTO DOS ATRIBUTOS ===============")
"""
O objetivo é deixar os valores perto um dos outros.
Por exemplo, em uma base de dados de usuários, a idade pode variar de 18 até 90 anos, o que é uma distância
muito grande de 18 e 90, e também tem o problema dos algoritmos de cálculos considerar 90 mais importante que 18.
"""

# Padronização (Standardisation) => É usado quando tem muito outliers (valores que estão muito fora do padrão)
scales_census = StandardScaler()
X_census = scales_census.fit_transform(X_census)
print(f"{X_census[0]}")

print("=============== DIVISÃO DAS BASES EM TREINAMENTO E TESTE ===============")
X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = train_test_split(
    X_census,
    y_census,
    test_size=0.15,
    random_state=0  # When you run code again, the "0" means to have the same split result between run code
)

print(f"{X_census_treinamento.shape=}")
print(f"{y_census_treinamento.shape=}")
print(f"{X_census_teste.shape=}, {y_census_teste.shape=}")

with open('census.pkl', mode='wb') as f:
    pickle.dump([X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste], f)
