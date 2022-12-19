import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(20.0,10.0)
#reading the data
data=pd.read_csv('headbrain.csv')
print(data.shape)
data.head()
#collecting the x and y
X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values
# calculating the mean
mean_x = np.mean(X)
mean_y = np.mean(Y)
#total no of values
n=len(X)
nume = 0
deno = 0
for i in range(n):
    nume = nume + (X[i] - mean_x) * (Y[i] - mean_y)
    deno = deno + (X[i] - mean_x)**2
m = nume / deno
c = mean_y - (m*mean_x)
print(m,c)

# Plotting Values and Regression Line
max_x = np.max(X) + 100
min_x = np.min(X) - 100
# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)
y = c + m * x

# Ploting Line
plt.plot(x, y, color='#52b920', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef4423', label='Scatter Plot')
plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

#r square method
numer_sq=0
deno_sq=0
for i in range(n):
    y_pred = c + m * X[i]
    numer_sq = (y_pred - mean_y)**2
    numer_sq = (Y[i] - mean_y) ** 2

r2 = 1 - (numer_sq / deno_sq)
print(r2)