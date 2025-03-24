import pandas as pd
from apyori import apriori

# header=None => the csv don't have a header, first row is not the header
base_mercado_1 = pd.read_csv('../Bases de dados/mercado.csv', header=None)

transacoes = []

for i in range(len(base_mercado_1)):
    transacoes.append([str(item) for item in base_mercado_1.iloc[i] if pd.notnull(item)])


print(transacoes)

regras = apriori(transacoes, min_support=0.3, min_confidence=0.8, min_lift=2)
resultados = list(regras)
print(resultados)
print(len(resultados))
print(resultados[0])

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
