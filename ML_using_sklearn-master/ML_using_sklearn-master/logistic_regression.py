#1st we have to load the dataset and analyze it
import pandas as pd
from sklearn.datasets import load_breast_cancer
import numpy as np

cancer = load_breast_cancer()
print(cancer.keys())
# print(data.feature_names)

# 2nd we have to create the data frame

# we have to create the dataframe of ur dataset
columns = np.append(cancer.feature_names, 'target');
print("Features Column Size: " + str(np.size(columns)))

# Append target data to current data
data = np.column_stack((cancer.data, cancer.target))
print("Data Column Size: " + str(np.size(data) / 569))

# Create dataframe with keywords
df = pd.DataFrame(data=data,  columns=columns)

#3rd  than we have to create x ,y feature space vector
# extact from our dataframe we have created
X = df.drop('target',axis = 1)
Y = df.get('target')

#4th and than we apply the train test the data
from sklearn.model_selection import train_test_split
X_train,x_test,Y_train,y_test=train_test_split(X,Y,random_state = 0)

# 5th and than finally we have to select the model to train
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()

# train the model this fit model find coeff of w & b and use it in the formula
# to classify the data
log_reg.fit(X_train,Y_train)
print('score of our model : {:.3f}'.format(log_reg.score(x_test,y_test)))

# 1st we have to analyze the data
# 2nd we have to create the dataframe
# 3rd we have to create the X ,Y the feature space in X contain all the data
# and Y contain its output
# 4th we have to split our data into training & testing data
# 5th we have to select our model and fit our training data

