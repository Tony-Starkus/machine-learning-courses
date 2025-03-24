import plotly.express as px
from sklearn import datasets
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

X_random, y_random = datasets.make_moons(n_samples=1500, noise=0.09)

grafico = px.scatter(x=X_random[:,0], y=X_random[:,1])
grafico.show()

kmeans = KMeans(n_clusters=2)
rotulos = kmeans.fit_predict(X_random)

grafico = px.scatter(x=X_random[:, 0], y=X_random[:, 1], color=rotulos)
grafico.show()

hc = AgglomerativeClustering(n_clusters=2, linkage='ward')
rotulos = hc.fit_predict(X_random)

grafico = px.scatter(x=X_random[:, 0], y=X_random[:, 1], color=rotulos)
grafico.show()

db_scan = DBSCAN(eps=0.1)
rotulos = db_scan.fit_predict(X_random)

grafico = px.scatter(x=X_random[:, 0], y=X_random[:, 1], color=rotulos)
grafico.show()
