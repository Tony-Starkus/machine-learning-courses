import Orange
from Orange.classification import MajorityLearner
from Orange.evaluation.testing import TestOnTestData
from collections import Counter

base_credit = Orange.data.Table('../Bases de dados/credit_data_regras.csv')
print(f"{base_credit.domain}")

majority = MajorityLearner()

previsoes = TestOnTestData(base_credit, base_credit, [majority])
print(Orange.evaluation.CA(previsoes))

for registro in base_credit:
    print(registro.get_class())

print(Counter(str(registro.get_class()) for registro in base_credit))
