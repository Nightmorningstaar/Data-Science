import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

plt.figure()
plt.title('Sample regression problem with one input variable')
X_R1, Y_R1 = make_regression(n_samples = 100, n_features=1,
                            n_informative=1, bias = 150.0,
                            noise = 30, random_state=0)
plt.scatter(X_R1, Y_R1, marker= 'o', s=50)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, x_test, Y_train, y_test = train_test_split(X_R1, Y_R1,random_state = 0)
linr_reg = LinearRegression()
linr_reg.fit(X_train, Y_train)#the linear regression fit method acts
# as estimate the future values of co_efficient (w) or (m) or slope or
#and the bias term (b) or (c) or intercept

print('linear model coeff (w) or (m): {}'.format(linr_reg.coef_))
print('linear model intercept (b) or (c): {:.3f}'.format(linr_reg.intercept_))
print('R-squared score (training): {:.3f}'.format(linr_reg.score(X_train, Y_train)))
print('R-squared score (test): {:.3f}'.format(linr_reg.score(x_test, y_test)))