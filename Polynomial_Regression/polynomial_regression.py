import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#get train data
data = pd.read_csv('xy.csv')
#analyse the data
print(data.head())
# print(data.describe())
# plt.scatter(data['x'],data['y'])
# plt.show()

#add x square column
data['x_2'] = [x**2 for x in data['x']]
x = np.array(data.drop('y',axis=1))
y = np.array(data['y'])
m = x.shape[0]
#add x0 = 1
x = np.concatenate((np.ones((m,1)),x),axis=1)

#use normal equation to calculate theta
theta = np.linalg.inv(x.T@x)@x.T@y
print(theta)

#get test data
x_data = pd.read_csv('test.csv')
x_data['x_2'] = [x**2 for x in x_data['x']]
x_test = np.array(x_data.iloc[:])
m_test=x_test.shape[0]
x_test = np.concatenate((np.ones((m_test,1)),x_test),axis=1)
#test y data is wrong for the particular prediction
y_data = pd.read_csv('sample_submission.csv')
y_test = np.array(y_data['y'])

#plot the values predicted from train data and actual data
# y_get = x@theta
# plt.scatter(x[:,1],y_get)
# plt.scatter(data['x'],data['y'])
# plt.show()

#predict y from test data
y_pred = x_test@theta

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
