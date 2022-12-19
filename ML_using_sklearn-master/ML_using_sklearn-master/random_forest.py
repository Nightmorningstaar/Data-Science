#https://github.com/cagdasyigit/coursera-applied-machine-learning-with-python/blob/master/Module%2B2.py
import pandas as pd
# import numpy as np

fruits = pd.read_table('C:\\Users\\ASUS\\Desktop\\python_prac\\knn\\fruit_data_with_colors.txt')
feature_names_fruits = ['height', 'width', 'mass', 'color_score']
# columns = np.append(fruits.feature_names_fruits)
# data = np.column_stack((fruits.feature_names_fruits))
X_fruits = fruits[feature_names_fruits]
Y_fruits = fruits['fruit_label']
target_names_fruits = ['apple', 'mandarin', 'orange', 'lemon']
print(fruits.keys())

from sklearn.model_selection import train_test_split
X_train,x_test,Y_train,y_test=train_test_split(feature_names_fruits ,fruits.fruit_label ,random_state = 0)

import matplotlib.pyplot as plt
fig,subaxes = plt.subplot(6,1,figsize = (6,32))
title = 'Random Forest fruits dataset,pair wise settings'
pair_list = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
# we iterate through pairs of features columns in the dataset

from sklearn.ensemble import RandomForestClassifier
from python_prac.adspy_shared_utilities import plot_class_regions_for_classifier_subplot

for pair,axis in zip(pair_list,subaxes):
    # for each pair of features we call the fit method on that subset of the training data X1
    # using the labels Y1
    X1 = X_train[:,pair]
    Y1 = Y_train

    clf = RandomForestClassifier()
    clf.fit(X1,Y1)
    #visualize the training data and the random forest decision boundries
    plot_class_regions_for_classifier_subplot(clf,X1,Y1,None,None,title,axis
                                              ,target_names_fruits)
    axis.set_xlabel(pair[0])
    axis.set_ylabel(pair[1])

plt.tight_layout()
plt.show()
clf1 = RandomForestClassifier(n_estimators = 10,random_state = 0).fit(X_train,Y_train)

print('RAndom Forest fruit dataset ,default setting')
print('Accuracy of RF classifier on the training set:{:.2f}'.format(clf1.score(X_train,Y_train)))
print('Accuracy of RF classifier on the test set:{:.2f}'.format(clf1.score(x_test,y_test)))

# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# iris = load_iris()
#
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
# fig, subaxes = plt.subplots(6, 1, figsize=(6, 32))
#
# pair_list = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
# tree_max_depth = 4
#
# from sklearn.ensemble import RandomForestClassifier
# from python_prac.adspy_shared_utilities import plot_class_regions_for_classifier_subplot
# for pair, axis in zip(pair_list, subaxes):
#     X = X_train[:, pair]
#     y = y_train
#
#     clf = RandomForestClassifier(max_depth=tree_max_depth).fit(X, y)
#     title = 'Decision Tree, max_depth = {:d}'.format(tree_max_depth)
#     plot_class_regions_for_classifier_subplot(clf, X, y, None,
#                                               None, title, axis,
#                                               iris.target_names)
#
#     axis.set_xlabel(iris.feature_names[pair[0]])
#     axis.set_ylabel(iris.feature_names[pair[1]])
#
# plt.tight_layout()
# plt.show()


# l =[1,2,3,4]
# l1 = [4,5,6,7]
# for l3,l4 in zip(l,l1):
#     print(l3)
#     print(l4)