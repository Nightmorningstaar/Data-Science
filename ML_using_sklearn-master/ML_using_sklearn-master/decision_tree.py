from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
Y = iris.target

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
X = iris.data
Y = iris.target
X_train, x_test, Y_train, y_test = train_test_split(X,Y,
        test_size=0.3,random_state = 1) # 70% training and 30% test

from sklearn.tree import DecisionTreeClassifier
decision_clf = DecisionTreeClassifier()
decision_clf.fit(X_train,Y_train)

import numpy as np

print('Accuracy in training data of cf1 : {:.3f}'.format(decision_clf.score(X_train,Y_train)))
print('Accuracy in testing data of cf1 : {:.3f}'.format(decision_clf.score(x_test,y_test)))
print('cross validation score of cf1 : {:.3f}'.format(np.mean(cross_val_score(decision_clf,X,Y,cv = 5))))
print('\n')

# Setting max decision tree depth to help overfitting
decision_clf2 = DecisionTreeClassifier(max_depth = 3)
decision_clf2.fit(X_train,Y_train)
print('Accuracy in training data of cf2: {:.3f}'.format(decision_clf2.score(X_train,Y_train)))
print('Accuracy in testing data of cf2 : {:.3f}'.format(decision_clf2.score(x_test,y_test)))

print('cross validation score of cf2 : {:.3f}'.format(np.mean(cross_val_score(decision_clf2,X,Y,cv = 5))))

# Visualizing decision trees
from python_prac.adspy_shared_utilities import plot_decision_tree
import matplotlib.pyplot as plt
plt.figure(figsize = (5,4))
plot_decision_tree(decision_clf, iris.feature_names, iris.target_names)
plt.show()

# By using Stratified cross validation
# from sklearn.model_selection import cross_val_score,StratifiedKFold
# import numpy as np
#
# stratified_kfold = StratifiedKFold(n_splits = 5,random_state = 0)
# cv_score = cross_val_score(clf,X,Y,cv = stratified_kfold)
# print('Stratified kfold score in Decision tree classifier : ',cv_score)
# print('Stratified kfold Mean score in Decision tree classifier : {:.3f}'.format(np.mean(cv_score)))

# Pre-pruned version (max_depth = 3)
# plot_decision_tree(decision_clf2, iris.feature_names, iris.target_names)
# plt.tight_layout()

from sklearn.datasets import load_breast_cancer
from python_prac.adspy_shared_utilities import plot_decision_tree

cancer = load_breast_cancer()
X_train, x_test, Y_train, y_test = train_test_split(cancer.data,cancer.target,
        test_size=0.3,random_state = 1) # 70% training and 30% test
clf1 = DecisionTreeClassifier(max_depth = 4,min_samples_leaf = 8,random_state = 0).fit(X_train,Y_train)
plot_decision_tree(clf1,cancer.feature_names,cancer.target_names)
plt.show()

# plot the important features of this data which is very helpful to categorize the data
from python_prac.adspy_shared_utilities import plot_feature_importances
plt.figure(figsize = (10,4))
plot_feature_importances(decision_clf2,feature_names = iris.feature_names)
plt.show()
print('feature importance : {}'.format(decision_clf2.feature_importances_))