# load the dataset
# our 1st step is to convert our dataset into dataframes
import  numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
fruits = pd.read_table('C:\\Users\\ASUS\\Desktop\\python_prac\\knn\\fruit_data_with_colors.txt')
print(fruits.head())
look_fruit_name = dict(zip(fruits.fruit_label.unique(),fruits.fruit_name.unique()))
print(look_fruit_name)

#create the train test split
# it will sspilt the data set into 75-25% of testing and trainning the data
X = fruits[['mass','width','height']]
Y = fruits.get(['fruit_label'])
X_train,x_test,Y_train,y_tests = train_test_split(X,Y, random_state = 0)

# create classifier object
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

# train the classifier(fit the estimator)using the training data
knn.fit(X_train,Y_train)#all estimators have the fit method that takes the training
# data and then changes the state of the classifier in other word it update the
# the state of the k and n variable here
# which means of knn it will memorize the training set example for future use
# Fit the model using X as training data and y as target values or o/p

# Estimate the accuracy of the classifier on future data using the test data
print(knn.score(x_test,y_tests))

# use the trained knn classifier model to classify new,priviously unseen objects
fruit_prediction = knn.predict([[20,4.3,5.5]])
print(look_fruit_name[fruit_prediction[0]])
# dict = {1:'praveen', 2:'bittoo'}
# print(dict[1])
fruit_prediction = knn.predict([[100,6.3,8.5]])
print(look_fruit_name[fruit_prediction[0]])

#Evaluation method
example_fruit = [[5.5,2.2,10]]
print('prdicted type of  ',example_fruit,'is',look_fruit_name[knn.predict(example_fruit)[0]])


# how sensitive  is knn  classification accuracy to the choice of the k perameters
k_range = range(1,20)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,Y_train)
    scores.append(knn.score(x_test,y_tests))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
# plt.legend()
plt.scatter(k_range,scores)
# plt.xticks([0,5,10,15,20]);
plt.show()