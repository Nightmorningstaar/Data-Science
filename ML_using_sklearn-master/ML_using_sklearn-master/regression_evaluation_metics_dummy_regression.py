from sklearn.datasets import load_diabetes

daibeties = load_diabetes()
X = daibeties.data[:,None,6]
Y = daibeties.target

from sklearn.model_selection import train_test_split

X_train, x_test, Y_train, y_test = train_test_split(X,Y,
        test_size=0.3,random_state = 1) # 70% training and 30% test

from sklearn.linear_model import LinearRegression
l_clf = LinearRegression()
l_clf.fit(X_train,Y_train)
y_linear_predict = l_clf.predict(x_test)


from sklearn.dummy import DummyRegressor
d_clf = DummyRegressor(strategy = 'mean')
d_clf.fit(X_train,Y_train)
y_dummy_predict = d_clf.predict(x_test)

print('Linear model coefficient :',l_clf.coef_)
print('Linear model intercept :',l_clf.intercept_)
print('Dummy model constant :',d_clf.constant_)

from sklearn.metrics import mean_squared_error,r2_score

print('Mean squared error for Linear model : {:.2f}'.format(mean_squared_error(y_true = y_test
                                                                               ,y_pred = y_linear_predict)))
print('r2 score for Linear model : {:.2f}'.format(r2_score(y_pred = y_linear_predict,y_true = y_test)))
print('Mean squared error for Dummy model : {:.2f}'.format(mean_squared_error(y_true = y_test,
                                                                              y_pred=y_dummy_predict)))
print('r2 score for Dummy model : {:.2f}'.format(r2_score(y_pred = y_dummy_predict,y_true = y_test)))

import matplotlib.pyplot as plt
plt.scatter(x_test,y_test,color = 'black')
plt.plot(x_test,y_linear_predict,color = 'green',linewidth = 2)
plt.plot(x_test,y_dummy_predict,color = 'red',linestyle = 'dashed',linewidth = 2,label = 'dummy')
plt.legend()
plt.show()