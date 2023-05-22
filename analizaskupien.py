import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from scipy.cluster.hierarchy import dendrogram, linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

data = pd.read_csv('countries of the world.csv', decimal=',')
print(data.head)
print(data.info())
data.fillna(0, inplace=True)
print(data.head)
print(data.info())
data_numeric= data.drop(['Country','Region'], axis=1)
label_encoder = LabelEncoder()
data['Country'] = label_encoder.fit_transform(data['Country'])
data['Region'] = label_encoder.fit_transform(data['Region'])

scaler = StandardScaler()
data_numeric = scaler.fit_transform(data_numeric)
data_numeric = pd.DataFrame(data_numeric)
data_numeric = pd.DataFrame(data_numeric, columns=data_numeric.columns)

X = data.iloc[:, 1:].values
print(data.head)
print(data.info())

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)


inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)


plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Liczba klastrów (k)')
plt.ylabel('Wartość inertia')
plt.title('Metoda łokcia ')
plt.show()


linkage_matrix = linkage(X, method='ward')
dendrogram(linkage_matrix, truncate_mode='level', p=4)
plt.xlabel('Indeksy próbek')
plt.ylabel('Odległość')
plt.title('Dendrogram')
plt.show()

