import Orange
from Orange.classification import MajorityLearner
from Orange.evaluation.testing import TestOnTestData
from collections import Counter

base_census = Orange.data.Table('../Bases de dados/census_regras.csv')
print(f"{base_census.domain}")

majority = MajorityLearner()

previsoes = TestOnTestData(base_census, base_census, [majority])
print(Orange.evaluation.CA(previsoes))

print(Counter(str(registro.get_class()) for registro in base_census))
