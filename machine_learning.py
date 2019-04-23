import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def warn_exer(i):
     return np.identity(i)
print(warn_exer(5))

data =   np.loadtxt("ex1data1.txt",delimiter=',')

X= np.c_[np.ones(data.shape[0]), data[:, 0]]
#print ("the value of x",data[:, 0])
"""np.ones(data.shape[0]) will give the outpt the value of x  the value of x [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1.]"""
"""
data[:, 0]] the value of x [ 6.1101  5.5277  8.5186  7.0032  5.8598  8.3829  7.4764  8.5781  6.4862
  5.0546  5.7107 14.164   5.734   8.4084  5.6407  5.3794  6.3654  5.1301
  6.4296  7.0708  6.1891 20.27    5.4901  6.3261  5.5649 18.945  12.828
 10.957  13.176  22.203   5.2524  6.5894  9.2482  5.8918  8.2111  7.9334
  8.0959  5.6063 12.836   6.3534  5.4069  6.8825 11.708   5.7737  7.8247
  7.0931  5.0702  5.8014 11.7     5.5416  7.5402  5.3077  7.4239  7.6031
  6.3328  6.3589  6.2742  5.6397  9.3102  9.4536  8.8254  5.1793 21.279
 14.908  18.959   7.2182  8.2951 10.236   5.4994 20.341  10.136   7.3345
  6.0062  7.2259  5.0269  6.5479  7.5386  5.0365 10.274   5.1077  5.7292
  5.1884  6.3557  9.7687  6.5159  8.5172  9.1802  6.002   5.5204  5.0594
  5.7077  7.6366  5.8707  5.3054  8.2934 13.394   5.4369]
"""
y = np.c_[data[:,1]]
plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()


def computeCost(X, y, theta=[[0], [0]]):
     m = y.size
     print("valve of  m",m)
     J = 0

     h = X.dot(theta)
     print("valve of y",y)
     print("valve of np.square",np.square(h - y))
     """ h =0 and y  is output  and np.square  valv is sqaure of h-y """
     J = 1 / (2 * m) * np.sum(np.square (h - y))
     print("value of j",J)
     return (J)
print("valve of computeCost",computeCost(X,y))


def gradientDescent(X, y, theta=[[0], [0]], alpha=0.01, num_iters=1500):
     m = y.size
     J_history = np.zeros(num_iters)

     for iter in np.arange(num_iters):
          h = X.dot(theta)
          theta = theta - alpha * (1 / m) * (X.T.dot(h - y))
          J_history[iter] = computeCost(X, y, theta)
     return (theta, J_history)
# theta for minimized cost J
theta , Cost_J = gradientDescent(X, y)
print('theta: ',theta.ravel())

plt.plot(Cost_J)
plt.ylabel('Cost J')
plt.xlabel('Iterations')