import pandas as pd
import numpy as np
import matplotlib.colors as clrs
import matplotlib.pyplot as plt

data = pd.read_csv('microchip_tests.txt',header=None)
# print(data.head())
# print(data.describe())
# colors = ['yellow','blue']
# plt.scatter(data[0],data[1], c=data[2], cmap =clrs.ListedColormap(colors))
# plt.show()

data[3] = [x**2 for x in data[0]]
data[4] = [x**2 for x in data[1]]
data[5] = [x**3 for x in data[0]]
data[6] = [x**3 for x in data[1]]
data[7] = [x*y for (x,y) in zip(data[0],data[1])]
data[8] = [x*(y**2) for (x,y) in zip(data[0],data[1])]
data[9] = [(x**2)*y for (x,y) in zip(data[0],data[1])]

m = len(data)

x = np.array(data.drop([2],axis=1))
x = np.concatenate((np.ones((m,1)),x),axis=1)
y = np.array(data[2]).reshape((m,1))

#train test split
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
lambd = 0.006
alph = 0.08

# print(x)
# print(y)

#sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

#hypothesis function
def hypothesis_function(theta):
    return sigmoid(x_train@theta)

#cost function vectorized implementation
def cost_function(theta):
    h_theta = hypothesis_function(theta)
    return (-1/m)*((y_train.T@np.log(h_theta)) + ((1-y_train).T@np.log(1-h_theta))) + np.sum(np.array([x**2 for x in theta]))*(lambd/m) - (theta[0]**2)*(lambd/m)

#gradient descent algorithm vectorized implementation
def gradient_descent(theta):
    prev_cost = cost_function(theta)
    for i in range(10000):
        temp = theta[0]
        theta = theta*(1-(lambd*alph)/m) - (alph/m)*x_train.T@(hypothesis_function(theta)-y_train)
        theta[0] = theta[0] + temp*(lambd*alph)/m
        cur_cost = cost_function(theta)
        if cur_cost>prev_cost:
            print(f"{i} error")
            break
        prev_cost = cur_cost
    return theta

theta = np.ones((x.shape[1],1))
#run gradient descent and get theta parameters
theta = gradient_descent(theta)
print(theta)

#test data and its size
m_test = x_test.shape[0]
#predicted y on test data
y_pred = sigmoid(x_test@theta)

# percentage of correct predictions
count=0
for i in range(m_test):
    #count number of correct predictions
    if (y_pred[i]>=0.5 and y_test[i]==1) or (y_pred[i]<0.5 and y_test[i]==0):
        count+=1
#print accurate prediction percentage
print(count*100/m_test)

xx = []
yy = []

xxxxx = np.arange(-1.0, 1.2, 0.01)

for i in xxxxx:
    for j in xxxxx:
        xx.append([i, j])
        x_ = np.array([1.0, i, j, i**2, j**2, i**3, j**3, i*j , i*(j**2) , (i**2)*j])
        y_ = sigmoid(x_ @ theta)
        if y_ >= 0.5:
            yy.append(1.0)
        elif y_ < 0.5:
            yy.append(0.0)

xx = np.array(xx)
yy = np.array(yy)

plt.scatter(xx[np.ravel(yy) == 1, 0], xx[np.ravel(yy) == 1, 1], c='grey')
plt.scatter(xx[np.ravel(yy) == 0, 0], xx[np.ravel(yy) == 0, 1], c='white')
plt.scatter(x[np.ravel(y) == 1, 1], x[np.ravel(y) == 1, 2], c='blue')
plt.scatter(x[np.ravel(y) == 0, 1], x[np.ravel(y) == 0, 2], c='orange')
plt.show()
