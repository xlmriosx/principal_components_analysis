import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


iris = sns.load_dataset('iris')

scaler = StandardScaler()
scaled = scaler.fit_transform(
    iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    )

covariance_matrix = np.cov(scaled.T)
print(covariance_matrix)

sns.pairplot(iris)
plt.show()

sns.jointplot(x= iris['petal_length'], y=iris['petal_width'])
plt.show()

sns.jointplot(x = scaled[:, 2], y = scaled[:,3])
plt.show()

eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
print(eigen_values)
print(eigen_vectors)

variance_explained = []
for i in eigen_values:
    variance_explained.append((i/sum(eigen_values))*100)

print(variance_explained)

pca = PCA(n_components=2)
pca.fit(scaled)

print(pca.explained_variance_ratio_)

reduced_scaled = pca.transform(scaled)
print(reduced_scaled)

iris['pca_1'] = scaled[:,0]
iris['pca_2'] = scaled[:,1]
sns.jointplot(iris['pca_1'], iris['pca_2'], hue = iris['species'])
plt.show()
