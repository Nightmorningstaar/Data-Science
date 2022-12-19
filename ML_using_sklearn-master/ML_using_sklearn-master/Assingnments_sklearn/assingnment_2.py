# Write a function that fits a polynomial LinearRegression model on the training data X_train for degrees 1, 3, 6, and 9.
# (Use PolynomialFeatures in sklearn.preprocessing to create the polynomial features and then fit a linear regression model)
# For each model, find 100 predicted values over the interval x = 0 to 10 (e.g. np.linspace(0,10,100)) and store this in a
# numpy array. The first row of this array should correspond to the output from the model trained on degree 1, the second
# row degree 3, the third row degree 6, and the fourth row degree 9.

import  numpy as np

n = 15

x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10

from sklearn.model_selection import train_test_split

X_train, x_test, Y_train, y_test = train_test_split(x, y, random_state=0)

def answer_one():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    result = np.zeros((4, 100))

    for i, degree in enumerate([1, 3, 6, 9], start=0):
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_train.reshape(11, 1))
        linreg = LinearRegression().fit(X_poly, Y_train)
        y = linreg.predict(poly.fit_transform(np.linspace(0, 10, 100).reshape(100, 1)))
        result[i, :] = y

    return result

#Write a function that fits a polynomial LinearRegression model on the training data X_train for degrees 0 through 9.
# For each model compute the  R2 (coefficient of determination) regression score on the training data as well as the
# the test data, and return both of these arrays in a tuple.This function should return one tuple of numpy
# arrays (r2_train, r2_test). Both arrays should have shape (10,)

def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    r2_train = np.zeros(10)
    r2_test = np.zeros(10)

    for i, degree in enumerate([0,1,2,3,4,5,6,7,8,9],start = 0):
        poly = PolynomialFeatures(degree = degree)
        X_train_poly = poly.fit_transform(X_train.reshape(11,1))
        x_test_poly = poly.fit_transform(x_test.reshape(4,1))
        clf = LinearRegression()
        clf.fit(X_train_poly,Y_train)
        r2_train[i] = clf.score(X_train_poly,Y_train)
        r2_test[i] = clf.score(x_test_poly, y_test)

    return (r2_train,r2_test)

def answer_three():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    r2_train = np.zeros(10)
    r2_test = np.zeros(10)

    for i, degree in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], start=0):
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train.reshape(11, 1))
        x_test_poly = poly.fit_transform(x_test.reshape(4, 1))
        clf = LinearRegression()
        clf.fit(X_train_poly, Y_train)
        r2_train[i] = clf.score(X_train_poly, Y_train)
        r2_test[i] = clf.score(x_test_poly, y_test)
        # print('r2 score training of degree :', i, 'score :', r2_train[i])
        # print('r2 score testing of degree :', i, 'score :', r2_test[i])
        # print('\n')

    overfitting = np.max(r2_train - r2_test)
    underfitting = np.min(r2_train)
    good_gen = np.min(r2_train - r2_test)

    r1 = np.where(overfitting == np.amax(overfitting))
    r2 = np.where(overfitting == np.amin(underfitting))
    r3 = np.where(overfitting == np.amin(good_gen))

    return (r2[0],r1[0],r3[0])

import pandas as pd

def answer_three1():
    r2_scores = answer_two()
    df = pd.DataFrame({'training_score': r2_scores[0], 'test_score': r2_scores[1]})
    df['diff'] = df['training_score'] - df['test_score']

    df = df.sort_values(['diff'])
    good_gen = df.index[0]

    df = df.sort_values(['diff'], ascending=False)
    overfitting = df.index[0]

    df = df.sort_values(['training_score'])
    underfitting = df.index[0]

    return (underfitting, overfitting, good_gen)

#Training models on high degree polynomial features can result in overly complex models that overfit, so we often use
# regularized versions of the model to constrain model complexity, as we saw with Ridge and Lasso linear regression.
#For this question, train two models: a non-regularized LinearRegression model (default parameters) and a regularized
# Lasso Regression model (with parameters alpha=0.01, max_iter=10000) both on polynomial features of degree 12.
# Return the  R2 score for both the LinearRegression and Lasso model's test sets.
#This function should return one tuple (LinearRegression_R2_test_score, Lasso_R2_test_score)

def answer_fourth():
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree = 12)
    X_train_poly = poly.fit_transform(X_train.reshape(11,1))
    x_test_poly = poly.fit_transform(x_test.reshape(4,1))

    from sklearn.linear_model import LinearRegression
    clf = LinearRegression()
    clf.fit(X_train_poly,Y_train)
    r2_test_lig =  clf.score(x_test_poly,y_test)

    from sklearn.linear_model import Lasso
    clf1 = Lasso(alpha=0.01, max_iter=10000)
    clf1.fit(X_train_poly,Y_train)
    r2_test_las = clf1.score(x_test_poly,y_test)

    return (r2_test_lig,r2_test_las)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('readonly/mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, x_test2, Y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = x_test2
y_subset = y_test2

# Using x_train2 and y_train2 from the preceeding cell, train a DecisionTreeClassifier with default parameters and
# random_state=0. What are the 5 most important features found by the decision tree?
# As a reminder, the feature names are available in the X_train2.columns property, and the order of the features in
# X_train2.columns matches the order of the feature importance values in the classifier's feature_importances_ property.
# This function should return a list of length 5 containing the feature names in descending order of importance.
# Note: remember that you also need to set random_state in the DecisionTreeClassifier.

def answer_five():
    from sklearn.tree import DecisionTreeClassifier

    tree_clf = DecisionTreeClassifier().fit(X_train2, y_train2)

    feature_names = []

    # Get index of importance leves since their's order is the same with feature columns
    for index, importance in enumerate(tree_clf.feature_importances_):
        # Add importance so we can further order this list, and add feature name with index
        feature_names.append([importance, X_train2.columns[index]])

    # Descending sort
    feature_names.sort(reverse=True)
    # Turn in to a numpy array
    feature_names = np.array(feature_names)
    # Select only feature names
    feature_names = feature_names[:5, 1]
    # Turn back to python list
    feature_names = feature_names.tolist()

    return feature_names

#For this question, we're going to use the validation_curve function in sklearn.model_selection to determine training and
# test scores for a Support Vector Classifier (SVC) with varying parameter values. Recall that the validation_curve function,
# in addition to taking an initialized unfitted classifier object, takes a dataset as input and does its own internal
# train-test splits to compute results.
#Because creating a validation curve requires fitting multiple models, for performance reasons this question will use
# just a subset of the original mushroom dataset: please use the variables X_subset and y_subset as input to the validation
# curve function (instead of X_mush and y_mush) to reduce computation time.
#The initialized unfitted classifier object we'll be using is a Support Vector Classifier with radial basis kernel.
# So your first step is to create an SVC object with default parameters (i.e. kernel='rbf', C=1) and random_state=0.
# Recall that the kernel width of the RBF kernel is controlled using the gamma parameter.
#With this classifier, and the dataset in X_subset, y_subset, explore the effect of gamma on classifier accuracy by using the
# validation_curve function to find the training and test scores for 6 values of gamma from 0.0001 to 10
# (i.e. np.logspace(-4,1,6)). Recall that you can specify what scoring metric you want validation_curve to use by setting
# the "scoring" parameter. In this case, we want to use "accuracy" as the scoring metric.
#For each level of gamma, validation_curve will fit 3 models on different subsets of the data, returning two 6x3
# (6 levels of gamma x 3 fits per level) arrays of the scores for the training and test sets.
#Find the mean score across the three models for each level of gamma for both arrays, creating two arrays
# of length 6, and return a tuple with the two arrays.
#e.g.

#if one of your array of scores is

#array([[ 0.5,  0.4,  0.6],
       #[ 0.7,  0.8,  0.7],
       #[ 0.9,  0.8,  0.8],
       #[ 0.8,  0.7,  0.8],
       #[ 0.7,  0.6,  0.6],
       #[ 0.4,  0.6,  0.5]])
#it should then become

#array([ 0.5,  0.73333333,  0.83333333,  0.76666667,  0.63333333, 0.5])
#This function should return one tuple of numpy arrays (training_scores, test_scores) where each array in the tuple has shape (6,).


def answer_six():
    from sklearn.svm import SVC
    clf = SVC(kernel = 'rbf',C = 1,random_state = 0)
    gamma = np.logspace(-4,1,6)

    from sklearn.model_selection import validation_curve
    train_scores, test_scores = validation_curve(clf,X_subset,y_subset,param_name = gamma,param_range = gamma
                                                 ,scoring = 'accuracy')
    return (test_scores,test_scores)
