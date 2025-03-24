import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

base_salario = np.array([
     [20, 1000],
     [27, 1200],
     [21, 2900],
     [37, 1850],
     [46, 900],
     [53, 950],
     [55, 2000],
     [47, 2100],
     [52, 3000],
     [32, 5900],
     [39, 4100],
     [41, 5100],
     [39, 7000],
     [48, 5000],
     [48, 6500]
])

# Escalonando/normalizando valores
scaler_salario = StandardScaler()
base_salario = scaler_salario.fit_transform(base_salario)


grafico1 = px.scatter(x=base_salario[:, 0], y=base_salario[:, 1])
grafico1.show()


plt.switch_backend('TkAgg')
plt.figure(figsize=(10, 5))
dendograma = dendrogram(linkage(base_salario, method='ward'))
plt.title('Dendograma')
plt.xlabel('Pessoas')
plt.ylabel('Distância')
plt.show()

hc_salario = AgglomerativeClustering(n_clusters=3, linkage='ward')
rotulos = hc_salario.fit_predict(base_salario)
print(rotulos)

grafico = px.scatter(x=base_salario[:, 0], y=base_salario[:, 1], color=rotulos)
grafico.show()
