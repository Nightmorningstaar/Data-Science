import numpy as np
arr = np.array([[56.0, 0.0, 4.4, 68.0],
             [1.2, 104.0, 52.0, 8.0],
             [1.8, 135.0, 99.0, 0.9]])

# axis hyperparameter can decide the way of sum '0' means colomn major & '1' means row major
calc = arr.sum(axis = 0)
# print(100 * ( arr / calc))# for division coloumn must be same
# print(100 * (np.array([[56.0, 0.0, 4.4, 68.0],
#              [1.2, 104.0, 52.0, 8.0],
#              [1.8, 135.0, 99.0, 0.9]]) / arr.sum(axis = 0) ))


# Rank 1 array we dont use that array for implementing LR and NN
a = np.random.randn(5)
print(a)
print(a.shape)# rank 1 array
print(a.T)
print(np.dot(a,a.T))

# We use this to implement LR and NN
a = np.random.randn(5,1)
print(a)
print(a.T)
print(np.dot(a,a.T))

arr = np.array([
    [1, 2, 3, 4]
])
print(arr + 100)

arr = np.array([
    [1, 2, 3],
     [4, 5, 6]
])
arr1 = np.array([100, 200, 300])
print(arr + arr1)

arr = np.random.randn(5)
arr.reshape(5,1)#convert into rank 2 array
assert (arr.shape == (5,1))# this is the assert statement for surity check that this vector is (5,1)

#DIFFERENCE BITWEEN  DOT AND MULTIPLY
# import numpy as np
# arr1 = np.array([[1,2,3]])
# arr2 = np.array([[1],[2],[3]])
# print(np.dot(arr1, arr2))#
# print(arr1 * arr2) # Simple Broadcasting
# print(np.squeeze([[16]]) + np.squeeze([[17]]))