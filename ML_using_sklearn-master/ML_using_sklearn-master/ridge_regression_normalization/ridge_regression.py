import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 12})
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

boston=load_boston()
print(boston.keys())
print(boston.feature_names)
boston_df=pd.DataFrame(boston.data,columns=boston.feature_names)
boston_df['Price']=boston.target
X = boston_df.drop('Price',axis=1)
# print(X[0:3]) # check
Y = boston_df['Price']

X_train,x_test,Y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=3)

liner_reg = LinearRegression()
liner_reg.fit(X_train,Y_train)

liner_ridge = Ridge(alpha = 20.0)
liner_ridge.fit(X_train,Y_train)

print('score of liner_regression : {:.3f}'.format(liner_reg.score(x_test,y_test)))
print('score of ridge_regression : {:.3f}'.format(liner_ridge.score(x_test,y_test)))

from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()# check the op with this and without this and you can find how good
# is pre-processing

