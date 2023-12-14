import Orange
from Orange.classification.rules import CN2Learner

base_credit = Orange.data.Table('../Bases de dados/credit_data_regras.csv')
print(f"{base_credit.domain=}")

# 25% para testar e 75% para treinar
base_dividida = Orange.evaluation.testing.sample(base_credit, n=0.25)
print(f"{base_dividida[0]=}")
print(f"{base_dividida[1]=}")

base_teste = base_dividida[0]
base_treinamento = base_dividida[1]
print(f"{len(base_treinamento)=} | {len(base_teste)=}")

cn2 = CN2Learner()
regras_credit = cn2(base_treinamento)

for regra in regras_credit.rule_list:
    print(f"regra: {regra}")

previsoes = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [lambda testdata: regras_credit])

print(f"{previsoes=}")

print(f"accuracy: {Orange.evaluation.CA(previsoes)}")
