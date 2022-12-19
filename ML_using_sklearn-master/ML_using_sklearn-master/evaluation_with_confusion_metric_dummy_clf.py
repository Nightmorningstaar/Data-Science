import numpy as np
from sklearn.datasets import load_digits

datasets = load_digits()
X,Y = datasets.data,datasets.target
print(datasets.keys())
# print(datasets.target)
# print(datasets.target_names)
# print(datasets.data)
for target_names,class_count in zip(datasets.target_names,np.bincount(datasets.target)):
    print(target_names,class_count)

# We have to find the  positive or negative data if it is not 1 or 0 than it is neg if 1 than positive
# in binary classification

Y_binary_imbalaced = Y.copy()
Y_binary_imbalaced[Y_binary_imbalaced != 1]  = 0
print('Original labels :',Y[:30])
print('New binary labels',Y_binary_imbalaced[:30])
print(np.bincount(Y_binary_imbalaced))

from sklearn.model_selection import train_test_split

# Apply normal classifier
X_train, x_test, Y_train, y_test = train_test_split(X,Y_binary_imbalaced,
        test_size=0.3,random_state = 1) # 70% training and 30% test

from sklearn.svm import SVC
clf = SVC(kernel = 'rbf',C = 1)
clf.fit(X_train,Y_train)
print('Accuracy of normal classifier :',clf.score(x_test,y_test))

# Dummy classifier
from sklearn.dummy import  DummyClassifier
dummy_majority = DummyClassifier(strategy = 'most_frequent')
dummy_majority.fit(X_train,Y_train)
y_dummy_pred = dummy_majority.predict(x_test)
print(y_dummy_pred)
print('Accuracy of dummy classifier :',dummy_majority.score(x_test,y_test))

from sklearn.dummy import DummyClassifier
clf_dummy = DummyClassifier(strategy = 'stratified').fit(X_train,Y_train)
print('Score Dummy stratified : {:.3f}'.format(clf_dummy.score(x_test,y_test)))

from sklearn.dummy import DummyClassifier
clf_dummy = DummyClassifier(strategy = 'uniform').fit(X_train,Y_train)
print('Score Dummy uniform: {:.3f}'.format(clf_dummy.score(x_test,y_test)))

# Confusion matrix
from sklearn.metrics import confusion_matrix
metric = confusion_matrix(y_pred = y_dummy_pred,y_true = y_test)
print('Confusion Metric of dummy clf :\n',metric)

# Confusion matrix
# from sklearn.metrics import  confusion_matrix
# dummy_majority2 = DummyClassifier(strategy = 'most_frequent')
# dummy_majority2.fit(X_train,Y_train)
# y_pred_majority = dummy_majority2.predict(x_test)
# confusion_metric = confusion_matrix(y_true = y_test,y_pred = y_pred_majority)# parameters  : Y_true_test_set,Y_pred_set
# print('Confusion Mertic of dummy clf \n',confusion_metric)


from sklearn.tree import DecisionTreeClassifier
decision_clf = DecisionTreeClassifier(max_depth = 2)
decision_clf.fit(X_train,Y_train)
y_decision_pred = decision_clf.predict(x_test)
metric2 = confusion_matrix(y_pred = y_decision_pred,y_true = y_test)
print('Confusion Metric of decision tree clf :\n',metric2)

# Evaluation metrices for binary classifiaction
from sklearn.metrics import recall_score,accuracy_score,precision_score,f1_score
print('Accuracy : {:.2f}'.format(accuracy_score(y_true = y_test,y_pred = y_decision_pred)))
print('Precision : {:.2f}'.format(precision_score(y_true = y_test,y_pred = y_decision_pred)))
print('Recall : {:.2f}'.format(recall_score(y_true = y_test,y_pred = y_decision_pred)))
print('F1 score : {:.2f}'.format(f1_score(y_true = y_test,y_pred = y_decision_pred)))


# By making a report to visualize the data in table form
from sklearn.metrics import classification_report
print('Classification report of our decision tree :',classification_report(y_true= y_test,
                                                    y_pred = y_decision_pred,target_names = ['not 1','1']))

#Decision Function
y_scores_svm = clf.fit(X_train,Y_train).decision_function(x_test)
y_scores_list = list(zip(y_test[:30],y_scores_svm[:30]))
for i in y_scores_list:
        print(i)