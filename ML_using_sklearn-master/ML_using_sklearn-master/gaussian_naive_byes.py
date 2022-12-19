from sklearn.datasets import make_classification

X_C2,Y_C2 = make_classification(n_samples = 100,n_informative = 2,n_features = 2,n_redundant = 0,
                                n_clusters_per_class = 1,flip_y = 0.1
                                ,class_sep = 0.5,random_state = 0)

import matplotlib.pyplot as plt
# plt.scatter(X_C2[1:20],Y_C2[1:20])
# plt.show()
x = [1,2,3,4,5,6,6,7]
y = [5,3,1,2,5,6,8,5]
plt.scatter(x,y,label = 'scatter',color = 'k')
plt.show()

from sklearn.model_selection import train_test_split

X_train, x_test, Y_train, y_test = train_test_split(X_C2 , Y_C2,random_state = 0)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train,Y_train)

from python_prac.adspy_shared_utilities import plot_class_regions_for_classifier
plot_class_regions_for_classifier(clf,X_train,Y_train,x_test,y_test,'gaussian naive byes clf')
print(clf.score(x_test,y_test))