import pandas as pd
from apyori import apriori

# header=None => the csv don't have a header, first row is not the header
base_mercado_2 = pd.read_csv('../Bases de dados/mercado2.csv')

transacoes = []

for i in range(base_mercado_2.shape[0]):
    print(i, range(base_mercado_2.shape[0]))
    transacoes.append([base_mercado_2.values[i, j] for j in range(base_mercado_2.shape[1]) if pd.notnull(base_mercado_2.values[i, j])])


# Produtos que são vendidos 4 vezes por dia
regras = apriori(transacoes, min_support=0.003, min_confidence=0.2, min_lift=2)
resultados = list(regras)
print(resultados)
print(len(resultados))

A = [] # SE
B = [] # ENTÃO
suporte = []
confianca = []
lift = []

for resultado in resultados:
    s = resultado[1]
    result_rules = resultado[2]
    for result_rule in result_rules:
        a = list(result_rule[0])
        b = list(result_rule[1])
        c = result_rule[2]
        l = result_rule[3]
        A.append(a)
        B.append(b)
        suporte.append(s)
        confianca.append(c)
        lift.append(l)


rules_df = pd.DataFrame({'A': A, 'B': B, 'suporte': suporte, 'confianca': confianca, 'lift': lift})
print(rules_df.sort_values(by='lift', ascending=False))
