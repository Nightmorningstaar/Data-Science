# import pandas as pd
# from sklearn.model_selection import cross_val_score
# from sklearn.neighbors import KNeighborsRegressor
# # from python_prac.knn.knn_dataset import fruits
# # from python_prac.adspy_shared_utilities import *
# knn = KNeighborsRegressor(n_neighbors = 5)
# fruits_data = pd.read_table('C:\\Users\\ASUS\\Desktop\\python_prac\\'
#                            'fruit dataset\\fruit_data_with_colors.txt')
#
# X_fruits_data_2d = fruits_data[['mass','width','height']]
# Y_fruits_data_2d = fruits_data['fruit_label']
#
# X = X_fruits_data_2d.to_numpy()
# Y = Y_fruits_data_2d.to_numpy()
# cv_score = cross_val_score(knn,X,Y)# estimator : estimator object implementing 'fit'
# # The object to use to fit the data.
# print('Cross_validadtion score (3-fold):',cv_score)
# import numpy as np
# print('Mean cross validation score (3-fold): {:.3f}',np.mean(cv_score))

from sklearn.datasets import load_iris
X,Y = load_iris(return_X_y = True)

from sklearn.svm import SVC
clf = SVC()

#StratifiedKFold cross validation
from sklearn.model_selection import StratifiedKFold,cross_val_score

stratified_kfold = StratifiedKFold(n_splits = 5,random_state = 0)
cv_score = cross_val_score(clf,X,Y,cv = stratified_kfold)
print("Stratified cross validation score :",cv_score)

import numpy as np

print("Mean Stratified cross validation score :",np.mean(cv_score))

#Cross validation
from sklearn.model_selection import KFold,cross_val_score

kfold = KFold(n_splits = 5,random_state = 0)
cv_score = cross_val_score(clf,X,Y,cv = kfold)
print("cross validation score :",cv_score)

import numpy as np

print("Mean cross validation score :",np.mean(cv_score))
