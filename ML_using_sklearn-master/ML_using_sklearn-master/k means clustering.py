from sklearn.datasets import make_blobs

X,Y = make_blobs(random_state = 10)

# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters = 3)
# k_predict = kmeans.fit_predict(X)
#
# from python_prac.adspy_shared_utilities import plot_labelled_scatter
# plot_labelled_scatter(X,k_predict,class_labels = ['cluster1','cluster2','cluster3'])

# from sklearn.cluster import KMeans
#
# k_means = KMeans(n_clusters = 3)
# k_means.fit(X)

from python_prac.adspy_shared_utilities import plot_labelled_scatter

# plot_labelled_scatter(X,k_means.labels_,['cluster1','cluster2','cluster3'])

# import pandas as pd
#
# fruits = pd.read_table('C:\\Users\\ASUS\\Desktop\\python_prac\\knn\\fruit_data_with_colors.txt')
# X_fruits = fruits[['mass','width','height','color_score']].as_matrix()
# Y_fruits = fruits[['fruit_label']] - 1
#
# from sklearn.preprocessing import MinMaxScaler
# X_fruits_normalized = MinMaxScaler().fit_transform(X_fruits)
#
# k_means2 = KMeans(n_clusters = 4,random_state = 0)
# k_means2.fit(X_fruits)
#
# plot_labelled_scatter(X,k_means2.labels_,['cluster1','cluster2','cluster3','cluster4'])

# AGGLOMERATIVE CLUSTERING
from sklearn.cluster import AgglomerativeClustering
cls = AgglomerativeClustering(n_clusters = 3)#n_clusters paramter that causesthe algo to stop when it has
#reach the number of clusters

# cls.fit(X)
#cls.predict(X)
cls_predict = cls.fit_predict(X)
plot_labelled_scatter(X = X,y = Y,class_labels = ['cluster1','cluster2','cluster3'])

#DENDOGRAM EXAPMPLE(IT IS USED TO FOR REPRESNTATION OF HEIRARICHAL CLUSTERING
from scipy.cluster.hierarchy import ward,dendrogram
# these ward function returns an array which can then be passed to the dendogram function to plot the tree
X,Y = make_blobs(random_state = 10,n_samples = 10)

import matplotlib.pyplot as plt
plt.figure()
dendrogram(ward(X))
plt.title('Dendogram')
plt.show()

from sklearn.cluster import DBSCAN

X1,Y1 = make_blobs(random_state = 9,n_samples = 25)
dbscan = DBSCAN(eps = 2,min_samples = 2)
cls = dbscan.fit_predict(X1)
print('cluster membership values : \n{}',format(cls))
print(cls)
plot_labelled_scatter(X1,cls + 1,['noise','cluster1','cluster2','cluster3'])
