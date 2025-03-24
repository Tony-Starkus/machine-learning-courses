import pandas as pd
from pyECLAT import ECLAT

# header=None => the csv don't have a header, first row is not the header
base_mercado_1 = pd.read_csv('../Bases de dados/mercado.csv', header=None)

eclat = ECLAT(data=base_mercado_1)
print(eclat.df_bin)
print(eclat.uniq_)

indices, suporte = eclat.fit(min_support=0.3, min_combination=1, max_combination=3)

print(f"{indices=}")
print(f"{suporte=}")
