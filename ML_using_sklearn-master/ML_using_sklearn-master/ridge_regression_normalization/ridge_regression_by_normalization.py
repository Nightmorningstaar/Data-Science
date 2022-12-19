import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 12})
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

boston=load_boston()
boston_df=pd.DataFrame(boston.data,columns=boston.feature_names)
boston_df['Price']=boston.target
X = boston_df.drop('Price',axis=1)
# print(X[0:3]) # check
Y = boston_df['Price']

X_train,x_test,Y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=3)

# we use the normaliztaion to reliable the data and remove noise and missing values
# to increase the accuracy
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# we have to fit the data before the transform
scaler.fit(X_train)

# we have to remove and missing values in training and testing data set
X_train_scaled = scaler.transform(X_train)
x_test_scaled = scaler.transform(x_test)

liner_ridge = Ridge(alpha = 20.0)
liner_ridge.fit(X_train_scaled,Y_train)

# print('score of liner_regression : {:.3f}'.format(liner_reg.score(x_test,y_test)))
print('score of ridge_regression : {:.3f}'.format(liner_ridge.score(x_test_scaled,y_test)))

# wtf normalize ke baad accuracy gir Q gyi bc...........






