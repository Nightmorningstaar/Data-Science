# l = [1, 2, 3, 4, 5]
# l1 = []
# for i in range(len(l)):
#     if i == 0:
#       l1.append(l[0])
#
#     else:
#         no = l[i] / l[i - 1]
#         l1.append(no)
#
# print(l1)

import numpy as np
print(np.array([1, 2, 3, 4]))

# creates an array of size 10 and fills with random values
a = np.random.rand(1000000)
b = np.random.rand(1000000)

import time
t1 = time.time()
ans = np.dot(a,b)
t2 = time.time()
print(ans)
print('Time in required vectorization :'+str(1000*(t2 - t1))+'ms')

t1 = time.time()
z = 0
for i in range(1000000):
    z += a[i] * b[i]
print(z)
t2 = time.time()
print('Time in required for loop(non vectorization) :'+str(1000*(t2 - t1))+'ms')

# 2 Example
u = np.zeros(5)
v = [1, 2, 3, 4, 5]
import math as m
for i in range(5):
    u[i] = m.exp(v[i])
print('By using for loop method :', u)

u = np.exp(v)
print('By using vectorized method :', u)

# Implementing gradient descent
j = 0
dw1 = 0
dw2 = 0
db = 0
for i in range(10):
    z[i] = w