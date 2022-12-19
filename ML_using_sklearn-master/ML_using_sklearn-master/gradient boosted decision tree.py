from sklearn.datasets import make_blobs

X_D2,Y_D2 = make_blobs(n_samples = 100,n_features = 2,centers = 8,
                       cluster_std = 1.3,random_state = 4)
Y_D2 = Y_D2 % 2

from sklearn.model_selection import train_test_split

X_train, x_test, Y_train, y_test = train_test_split(X_D2 , Y_D2,random_state = 0)

import matplotlib.pyplot as plt
fig,subaxes = plt.subplots(1,1,figsize = (6,6))

from sklearn.ensemble import GradientBoostingClassifier
title = 'GBDT complex binary dataset  default settings'
clf = GradientBoostingClassifier().fit(X_train,Y_train)

from python_prac.adspy_shared_utilities import plot_class_regions_for_classifier_subplot
plot_class_regions_for_classifier_subplot(clf,X_train,Y_train,x_test,y_test,title,subaxes)
plt.show()

# grid_values = {'learning_rate':[0.1, 1.0, 10.0, 100.0],
#                'n_estimators':[100,200,300,400,500,600,1000]}
#
# from sklearn.model_selection import GridSearchCV
# g_clf = GridSearchCV(estimator = clf,param_grid = grid_values)
# g_clf.fit(X_train,Y_train)
# print('Best score',g_clf.best_score_)
# print('Best parameter',g_clf.best_params_)
