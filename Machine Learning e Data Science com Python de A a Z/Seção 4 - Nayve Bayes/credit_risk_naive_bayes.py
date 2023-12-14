import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

base_risco_credito = pd.read_csv("./risco_credito.csv")

# Get all linhas the values: historia,divida,garantias,renda
X_risco_credito = base_risco_credito.iloc[:, 0:4].values

# Get all linhas the values of column risco
y_risco_credito = base_risco_credito.iloc[:, 4].values

label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantia = LabelEncoder()
label_encoder_renda = LabelEncoder()

X_risco_credito[:, 0] = label_encoder_historia.fit_transform(X_risco_credito[:, 0])
X_risco_credito[:, 1] = label_encoder_divida.fit_transform(X_risco_credito[:, 1])
X_risco_credito[:, 2] = label_encoder_garantia.fit_transform(X_risco_credito[:, 2])
X_risco_credito[:, 3] = label_encoder_renda.fit_transform(X_risco_credito[:, 3])

naive_risco_credito = GaussianNB()
naive_risco_credito.fit(X_risco_credito, y_risco_credito)

# história boa (0), divida alta (0), garantias nenhuma (1), renda > 35 (2)
# história ruim (2), divida alta (0), garantias adequada (0), renda < 15 (0)
previsao = naive_risco_credito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
print(f"{previsao=}")

print(f"{naive_risco_credito.classes_}")

# Quantity of cases for each class
print(f"{naive_risco_credito.class_count_}")

# Apriori probabilities
print(f"{naive_risco_credito.class_prior_}")
