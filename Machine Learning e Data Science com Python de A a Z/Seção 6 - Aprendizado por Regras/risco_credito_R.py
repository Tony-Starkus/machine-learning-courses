import Orange
from Orange.classification.rules import CN2Learner

base_risco_credito = Orange.data.Table('../Bases de dados/risco_credito_regras.csv')
print(f"{base_risco_credito=}")
print(f"{base_risco_credito.domain=}")

cn2 = CN2Learner()
regras_risco_credito = cn2(base_risco_credito)
print("==============")
for regras in regras_risco_credito.rule_list:
    print(regras)

# história boa, dívida alta, garantias nenhuma, renda > 35
# história ruim, dívida alta, garantias adequada, renda < 35

previsoes = regras_risco_credito([['boa', 'alta', 'nenhuma', 'acima_35'], ['ruim', 'alta', 'adequada', '0_15']])

print(f"{previsoes=}")
print(f"{base_risco_credito.domain.class_var.values=}")

for i in previsoes:
    print(base_risco_credito.domain.class_var.values[i])
