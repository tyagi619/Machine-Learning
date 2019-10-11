import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read data
data = pd.read_csv('data.csv')
#drop last column (unnamed column)
data = data.iloc[:,:-1]
data.drop('id',inplace=True,axis=1)
#change prediction class from string to int
cancer_type = {'M':1,'B':0}
data['diagnosis'] = [cancer_type[x] for x in data['diagnosis']]
#analyse data
# print(data.head())
# print(data.describe())

#length of dataset
m = len(data)

#get x data
x = np.array(data.drop('diagnosis',axis=1))
#mean normalization
for i in range(30):
    mean_x = np.mean(x[:][i])
    sd_x = np.std(x[:][i])
    for j in range(m):
        x[j][i] = (x[j][i]-mean_x)/sd_x
#add x0=1
x = np.concatenate((np.ones((m,1)),x),axis=1)
#get y data
y = np.array(data['diagnosis']).reshape((m,1))

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
    return (-1/m)*((y_train.T@np.log(h_theta)) + ((1-y_train).T@np.log(1-h_theta)))

#gradient descent algorithm vectorized implementation
def gradient_descent(theta):
    prev_cost = cost_function(theta)
    for i in range(1000):
        theta = theta - (0.3/m)*x_train.T@(hypothesis_function(theta)-y_train)
        cur_cost = cost_function(theta)
        if cur_cost>prev_cost:
            print(f"{i} error")
            break
        prev_cost = cur_cost
    return theta

theta = np.ones((31,1))
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
