import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read data
data = pd.read_csv('train.csv')
print(data.head())
data.dropna(inplace=True)

#training dataset
x_train= np.array(data['x'])
y_train= np.array(data['y'])
m = len(data['x'])

cost = []
#hypothesis function
def h_theta(theta_0,theta_1,x_i):
    return theta_0 + theta_1*x_i
#cost function
def cost_function(theta_0,theta_1):
    temp = 0;
    for i in range(m):
        temp = temp + (h_theta(theta_0,theta_1,x_train[i])-y_train[i])**2
    return temp/(2 * m)
#gradient descent algorithm
def gradient_descent(theta_0,theta_1):
    prev_cost = cost_function(theta_0,theta_1)
    cost.append(prev_cost)
    for i in range(10000):
        temp1 = 0.0
        temp2 = 0.0
        for j in range(m):
            temp1 = temp1 + (h_theta(theta_0,theta_1,x_train[j])-y_train[j])
            temp2 = temp2 + (h_theta(theta_0,theta_1,x_train[j])-y_train[j])*x_train[j]
        theta_0 = theta_0 - (0.0005)*(temp1)/m
        theta_1 = theta_1 - (0.0005)*(temp2)/m
        cur_cost = cost_function(theta_0,theta_1)
        cost.append(cur_cost)
        if cur_cost>prev_cost:
            print(i)
            break
        prev_cost = cur_cost
    return (theta_0,theta_1)

#value of theta
theta_0 = -2
theta_1 = 2
(theta_0,theta_1) = gradient_descent(theta_0,theta_1)
print(f"theta0 = {theta_0} and theta_1= {theta_1}")

#plot cost function vs number of iterations
# plt.plot(cost)
# plt.show()

#plot the obtained line and the dataset
# y_calc = []
# for i in range(999):
    # yVal = h_theta(theta_0, theta_1, x_train[i])
    # y_calc.append(yVal)

# plt.scatter(x_train, y_train)
# plt.plot(x_train, y_calc, 'r')
# plt.show()

#read test data
test_data = pd.read_csv('test.csv')
print(test_data.head())
test_data.dropna(inplace=True)
x_test = np.array(test_data['x'])
y_test = np.array(test_data['y'])

#predict y
y_pred = []
m_test = len(test_data['x'])
for i in range(m_test):
    y_pred.append(h_theta(theta_0,theta_1,x_test[i]))

#calculate relative mean squared error
#mean squared error
mse = 0
#variance
var = 0
mean_y = np.mean(y_test)
for i in range(m_test):
    mse = mse + (y_pred[i]-y_test[i])**2
    var = var + (y_test[i]-mean_y)**2

mse = mse/m_test
var = var/(m_test-1)
rMSE = mse/var
print(1-rMSE)
