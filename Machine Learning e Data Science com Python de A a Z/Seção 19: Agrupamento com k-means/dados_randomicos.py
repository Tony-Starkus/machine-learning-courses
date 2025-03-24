from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import plotly.express as px
import plotly.graph_objects as go

# 200 números aleatórios & 5 centroides
X_random, y_random = make_blobs(n_samples=200, centers=5, random_state=1)
print(f"{X_random=}")
print(f"{y_random=}")

grafico = px.scatter(x=X_random[:,0], y=X_random[:,1])
grafico.show()

kmeans_blobs = KMeans(n_clusters=5)
kmeans_blobs.fit(X_random)

rotulos = kmeans_blobs.predict(X_random)
centroides = kmeans_blobs.cluster_centers_
print(f"{centroides=}")

grafico1 = px.scatter(x=X_random[:,0], y=X_random[:,1], color=rotulos)
grafico2 = px.scatter(x=centroides[:,0], y=centroides[:,1], size=[5, 5, 5, 5, 5])
grafico3 = go.Figure(data=grafico1.data + grafico2.data)
grafico3.show()
