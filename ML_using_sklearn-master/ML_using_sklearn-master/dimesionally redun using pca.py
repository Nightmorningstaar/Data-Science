from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
(X_cancer,Y_cancer) = load_breast_cancer(return_X_y = True)#return_X_y : boolean, default=False
# If True, returns ``(data, target)``

# BEFORE USING PCA WE HAVE TO PREPROCESSING THE DATA SO THAT EACH FEATURES RANGE OF VALUES HAS
# ZERO MEAN AND UNIT VARIANCE
from sklearn.preprocessing import StandardScaler

# X_normalized = StandardScaler()
# X_normalized.fit(X_cancer)
# X_normalized.transform(X_cancer)
X_normalized = StandardScaler().fit_transform(X_cancer)
# THAN WE APPLY PCA TO DIMESIONALITY REDUCTION
from sklearn.decomposition import PCA
pca = PCA(n_components = 2).fit(X_normalized)# we reduce the dimesion of entire dataset into 2
X_pca = pca.transform(X_normalized)
print('Before reduction',X_cancer.shape)
print("After reduction",X_pca.shape)

# FOR VISUALIZATION
import matplotlib.pyplot as plt
from python_prac.adspy_shared_utilities import plot_labelled_scatter
# plot_labelled_scatter(X_pca,Y_cancer,['malignant','benign'])
# plt.xlabel('First principal componenet')
# plt.ylabel('Second principal component')
# plt.title('Breat Cancer Dataset PCA (n_componenets) = 2')

# IMPORT FRUIT DATASET
import pandas as pd

fruits = pd.read_table('C:\\Users\\ASUS\\Desktop\\python_prac\\knn\\fruit_data_with_colors.txt')
feature_names_fruits = ['height', 'width', 'mass', 'color_score']
# columns = np.append(fruits.feature_names_fruits)
# data = np.column_stack((fruits.feature_names_fruits))
X_fruits = fruits[feature_names_fruits]
Y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']

# BEFORE USING PCA WE HAVE TO PREPROCESSING THE DATA SO THAT EACH FEATURES RANGE OF VALUES HAS
# ZERO MEAN AND UNIT VARIANCE
X_normalized = StandardScaler().fit_transform(X_fruits)

# DIMENSION REDUN BY USING MDS(MULTI DIMENSIONAL SCALING) IN MANIFOLD LEARING ALGO
from sklearn.manifold import MDS

mds = MDS(n_components = 2)
X_normalized_mds = mds.fit_transform(X_normalized)
print('Before reduction in MDS',X_normalized.shape)
print('After reduction in MDS',X_normalized_mds.shape)
print('\n')

#FOR VISUALIZATION
# plot_labelled_scatter(X_normalized_mds,Y_fruits,['apple', 'mandarin', 'orange', 'lemon'])
# plt.xlabel('First MDS feature')
# plt.ylabel('Second  MDS feature')
# plt.title('Fruit sample Dataset MDS (n_componenets) = 2')

# DIMENSION REDUN BY USING t-SNE POWERFUL MANIFOLD LEARING ALGO
from sklearn.manifold import TSNE
tsne = TSNE(random_state = 0)
X_tsne = tsne.fit_transform(X_normalized)
print('Before reduction in t-SNE',X_normalized.shape)
print('After reduction in t-SNE',X_tsne.shape)


#FOR VISUALIZATION
plot_labelled_scatter(X_normalized_mds,Y_fruits,['apple', 'mandarin', 'orange', 'lemon'])
plt.xlabel('First t-SNE feature')
plt.ylabel('Second  t-SNE feature')
plt.title('Fruit sample Dataset t-SNE ')