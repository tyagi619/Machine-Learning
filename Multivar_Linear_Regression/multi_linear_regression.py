import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('insurance.csv')
print(data.head())
# print(data.describe())
# print(data['region'].nunique())
gender = {'male':1,'female':2}
smoker = {'yes':1,'no':0}
region = {'northeast':1,'southeast':2,'southwest':3,'northwest':4}
data['sex'] = [gender[x] for x in data['sex']]
data['smoker'] = [smoker[x] for x in data['smoker']]
data['region'] = [region[x] for x in data['region']]
# print(data.head())

x = np.array(data.iloc[:,:-1])
y = np.array(data['charges'])
m = x.shape[0]
#x0=1 for every data row
x = np.concatenate((np.ones((m,1)),x),axis=1)

cost=[]

#train test split function to split dataset
def train_test_split(x,y,s):
    val = int(m*(1-s))

    indices = np.arange(m)
    np.random.shuffle(indices)

    A = x[indices]
    B = y[indices]

    x_train = A[:val+1]
    y_train = B[:val+1]
    x_test = A[val+1:]
    y_test = B[val+1:]

    return (x_train,y_train,x_test,y_test)

(x_train,y_train,x_test,y_test) = train_test_split(x,y,0.3)
m = x_train.shape[0]

#hypothesis function
def h_theta(theta,index):
    y_val = 0
    for i in range(7):
        y_val += theta[i]*x_train[index][i]
    return y_val

#cost function
def cost_function(theta):
    temp = 0;
    for i in range(m):
        temp += (h_theta(theta,i)-y_train[i])**2
    return temp/(2 * m)

#gradient descent algorithm
def gradient_descent(theta):
    prev_cost = cost_function(theta)
    cost.append(prev_cost)
    for i in range(10000):
        temp = [0,0,0,0,0,0,0]
        for j in range(m):
            err = h_theta(theta,j)-y_train[j]
            for k in range(7):
                temp[k] += err*x_train[j][k]
        for j in range(7):
            theta[j] = theta[j] - 0.0006*temp[j]/m
        cur_cost = cost_function(theta)
        cost.append(cur_cost)
        if cur_cost>prev_cost:
            print(i)
            print("error")
            break
        prev_cost = cur_cost
    return theta

theta = [-12490,260,360,320,440,24000,-150]
theta = gradient_descent(theta)
#print final parameters
print(f"theta0 = {theta[0]}  theta1 = {theta[1]}  theta2 = {theta[2]}  theta3 = {theta[3]}  theta4 = {theta[4]}  theta5 = {theta[5]}  theta6 = {theta[6]}")

# plt.plot(cost)
# plt.show()

#hypothesis prediction on test data
def h_theta_predict(theta,index):
    y_val = 0
    for i in range(7):
        y_val += theta[i]*x_test[index][i]
    return y_val

#predict y
y_pred = []
m_test = x_test.shape[0]
for i in range(m_test):
    y_pred.append(h_theta_predict(theta,i))

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
print(var**0.5)
