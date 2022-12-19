#Import the data from fraud_data.csv. What percentage of the observations in the dataset are
# instances of fraud?This function should return a float between 0 and 1.
import pandas as pd
import numpy as np
data = pd.read_csv('C:\\Users\\ASUS\\Desktop\\python_prac\\fraud_data.csv')
se = pd.Series(np.bincount(data.Class))
print(se)
total = se[0] + se[1]
print('{:.3f}'.format((se[1] / total) * 100)+'%')

#Using X_train, X_test, y_train, and y_test (as defined above), train a dummy classifier that
# classifies everything as the majority class of the training data. What is the accuracy of
# this classifier? What is the recall?This function should a return a tuple with two floats
# , i.e. (accuracy score, recall score).

from sklearn.model_selection import train_test_split
df = pd.read_csv('C:\\Users\\ASUS\\Desktop\\python_prac\\fraud_data.csv')
X = df.iloc[:,:-1]
Y = df.iloc[:,-1]
X_train, x_test, Y_train, y_test = train_test_split(X, Y, random_state=0)

from sklearn.dummy import DummyClassifier
d_clf = DummyClassifier(strategy = 'most_frequent')
d_clf.fit(X_train,Y_train)

from sklearn.metrics import accuracy_score,recall_score,precision_score
y_pred = d_clf.predict(x_test)
print("Accuracy_Score of Dummy classifier : {:.3f}".format(accuracy_score(y_true = y_test,y_pred = y_pred)))
print("Recall_Score of Dummy classifier : {:.3f}".format(recall_score(y_true = y_test,y_pred = y_pred)))
print('\n')
#Using X_train, X_test, y_train, y_test (as defined above), train a SVC classifer using the default
# parameters. What is the accuracy, recall, and precision of this classifier?This function should
# a return a tuple with three floats, i.e. (accuracy score, recall score, precision score).

from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train,Y_train)
y_pred_clf = clf.predict(x_test)
print("Accuracy_Score of classifier : {:.3f}".format(accuracy_score(y_true = y_test,y_pred = y_pred_clf)))
print("Recall_Score of classifier : {:.3f}".format(recall_score(y_true = y_test,y_pred = y_pred_clf)))
print("Precision_Score of classifier : {:.3f}".format(precision_score(y_true = y_test,y_pred = y_pred_clf)))

#Using the SVC classifier with parameters {'C': 1e9, 'gamma': 1e-07}, what is the confusion matrix
# when using a threshold of -220 on the decision function. Use X_test and y_test.This function
# should return a confusion matrix, a 2x2 numpy array with 4 integers.

from sklearn.metrics import confusion_matrix
clf1 = SVC(C = 1e9,gamma = 1e-07)
clf1.fit(X_train,Y_train)
y_scores  = clf1.decision_function(x_test) > -220
mat = confusion_matrix(y_true = y_test,y_pred = y_scores)
print('confusion metrices :',mat)

# Perform a grid search over the parameters listed below for a Logisitic Regression classifier, using recall for scoring
# and the default 3-fold cross validation.
# 'penalty': ['l1', 'l2']
# 'C':[0.01, 0.1, 1, 10, 100]
# From .cv_results_, create an array of the mean test scores of each parameter combination. i.e.

# l1	l2
# 0.01	?	?
# 0.1	?	?
# 1	?	?
# 10	?	?
# 100	?	?

# This function should return a 5 by 2 numpy array with 10 floats.
# Note: do not return a DataFrame, just the values denoted by '?' above in a numpy array. You might need to reshape
# your raw result to meet the format we are looking for.

from sklearn.linear_model import LogisticRegression
clf3 = LogisticRegression()
clf3.fit(X_train,Y_train)
grid_values = {'C':[0.01, 0.1, 1, 10, 100]}

from sklearn.model_selection import GridSearchCV
grid_clf = GridSearchCV(clf3,param_grid = grid_values,scoring = 'recall')
grid_clf.fit(X_train,Y_train)
print(grid_clf.best_params_)
print(grid_clf.best_score_)
