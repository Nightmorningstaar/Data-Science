import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

plt.figure()
plt.title('Sample regression problem with one input variable')
X_R1, Y_R1 = make_regression(n_samples = 100, n_features=1,
                            n_informative=1, bias = 150.0,
                            noise = 30, random_state=0)
plt.scatter(X_R1, Y_R1, marker= 'o', s=50)
# plt.show()
#s means the size of the marker
#plt.show()

# from sklearn.datasets import make_friedman1
# plt.figure()
# plt.title('complex regression problem with one i/p vrbl')
# X_F1,Y_F1 = make_friedman1(n_samples = 100,n_features = 7,random_state = 0)
# plt.scatter(X_F1[:,2],Y_F1,marker = 'o', s = 50)
# plt.show()

from sklearn.datasets import make_classification
# from matplotlib.colors import ListedColormap
plt.figure()
# cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])
X_C2,Y_C2 = make_classification(n_samples = 100,n_features = 2,n_informative = 2,n_redundant = 0,
                                n_clusters_per_class = 1,flip_y = 0.1,class_sep = 0.5,
                                random_state = 0)
plt.scatter(X_C2[:, 0], X_C2[:, 1], marker= 'o',c = Y_C2, s = 50)#, cmap = cmap_bold)
plt.title('binary classification problem')
# plt.show()

# Classification
# from adspy_shared_utilities import plot_two_class_knn
# X_train,Y_train,x_test,y_test = train_test_split(X_C2,Y_C2,random_state = 0)
# plot_two_class_knn(X_train,Y_train,1,'uniform',x_test,y_test)
# plot_two_class_knn(X_train,Y_train,3,'uniform',x_test,y_test)
# plot_two_class_knn(X_train,Y_train,11,'uniform',x_test,y_test)

# Regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_R1, Y_R1, random_state = 0)
k = [1,2,3,4,5,6,7,8,9,10,11,55]

for i in k:
    knn_reg = KNeighborsRegressor(n_neighbors = i)
    knn_reg.fit(X_train, y_train)
    print(knn_reg.predict(X_test))
    print('R-squared test score: {:.3f}'.format(knn_reg.score(X_test, y_test)))
    print('\n')
