from sklearn.datasets import load_digits

dataset = load_digits()
X,Y = dataset.data,dataset.target == 1

# Cross validation
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

clf = SVC(kernel = 'linear',C = 1)
# The first call to cross val score just uses default accuracy as evaluation metric
print('Cross validation (accuracy)',cross_val_score(clf,X,Y,cv = 5))

# The second call uses the scoring parameter using roc_auc and this will use AUC as the evaluation metric
print('Cross validation (AUC)',cross_val_score(clf,X,Y,cv = 5,scoring = 'roc_auc'))

# The third call uses the scoring parameter to recall to use that as the evalution metric
print('Cross validation (Recall)',cross_val_score(clf,X,Y,cv = 5,scoring = 'recall'))

# The third call uses the scoring parameter to precision to use that as the evalution metric
print('Cross validation (precision)',cross_val_score(clf,X,Y,cv = 5,scoring = 'precision'))

# print('Mean cross validation score :', np.mean(cross_val_score(clf, X, Y, cv=5)))
# print('Mean Stratified validation score :', np.mean(cross_val_score(clf, X, Y, cv=StratifiedKFold(n_splits=5
#                                                                                      ,random_state = 0))))
#
# print('Stratified cross vaidation precision score :',np.mean(cross_val_score(clf,X,Y,cv = StratifiedKFold(
#                                                           n_splits = 5,random_state = 0),scoring = 'precision')))
#
# print('Stratified cross vaidation recall score :',np.mean(cross_val_score(clf,X,Y,cv = StratifiedKFold(
#                                                        n_splits = 5,random_state = 0),scoring = 'recall')))
#
# print('Stratified cross vaidation roc_auc score :',np.mean(cross_val_score(clf,X,Y,cv = StratifiedKFold(
#                                                         n_splits = 5,random_state = 0),scoring = 'roc_auc')))


# Grid Search example(used for selecting the particular dataset for selecting the hyper parameters)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  roc_auc_score
from sklearn.model_selection import train_test_split

X_train, x_test, Y_train, y_test = train_test_split(X,Y,
        test_size=0.3,random_state = 1) # 70% training and 30% test

# optimize for avg accuracy
clf2 = SVC(kernel = 'rbf')
grid_values = {'gamma':[0.001,0.01,0.05,0.1,1,10,20,30,50,100]}
grid_clf = GridSearchCV(clf2,param_grid = grid_values)
grid_clf.fit(X_train,Y_train)
# Decision fuction provide the test score of each and every point in the data set
y_decision_fn_scores = grid_clf.decision_function(x_test)

print('\n')
print('Grid best parameter (max accracy) : ',grid_clf.best_params_)
print('Grid best score (accuracy) : ',grid_clf.best_score_)
# print('Test case value :',y_decision_fn_scores)

#Optimize For roc_auc
grid_clf2 = GridSearchCV(clf2,param_grid = grid_values,scoring = 'roc_auc')
grid_clf2.fit(X_train,Y_train)
# Decision fuction provide the test score of each and every point in the data set
y_decision_fn_scores_auc = grid_clf.decision_function(x_test)

print('\n')
print('Grid best parameter (max accracy) : ',grid_clf2.best_params_)
print('Grid best score (accuracy) : ',grid_clf2.best_score_)
#print('Test case (AUC) :',y_decision_fn_scores_auc)

#EVALUATION MERTICS SUPPORTED FOR MODEL SELECTION
# from sklearn.metrics.scorer import _scorer
# sorted(list(_scorer.SCORERS))

#OPTIMIZING A CLASSIFIER USING DIFFERENT EVALUATION METRICES
import jitter as jitter
import numpy as np
import matplotlib.pyplot as plt
jitter_delta = 0.25
X_twovar_train = X_train[:,[20,59]] + np.random.rand(X_train.shape[0],2)-jitter_delta
x_twovar_test = x_test[:,[20,59]] + np.random.rand(x_test.shape[0],2)-jitter_delta
clf.fit(X_twovar_train,Y_train)
# AS THE WEIGHT PARAMETER INCREASES,MORE EMPHASIS WILL BE GIVEN TO CORRECTLY CLASSIFYING THE POSITIVE CLASS INSTANCES
grid_values2 = {'class_weight':['balanced',{1:2},{1:3},{1:4},{1:5},{1:10},{1:20},{1:50}]}
plt.figure()

from python_prac.adspy_shared_utilities import plot_class_regions_for_classifier

for i ,eval_matric in enumerate(('precision','recall','f1','roc_auc')):
        grid_clf_custum = GridSearchCV(clf2,param_grid = grid_values2,scoring = eval_matric)
        grid_clf_custum.fit(X_twovar_train,Y_train)
        print('Grid best parameter (max.{0}):{1}'.format(eval_matric,grid_clf_custum.best_params_))
        print('Grid best score ({0}):{1}'.format(eval_matric,grid_clf_custum.best_score_))
        plt.subplot(2,2,i+1)
        plt.subplots_adjust(wspace = 0.3,hspace = 0.3)
        plot_class_regions_for_classifier(grid_clf_custum,x_twovar_test,y_test)
        plt.title(eval_matric+'-oriented SVC')
plt.show()
