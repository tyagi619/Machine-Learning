import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read dataset
data = pd.read_csv('IRIS.csv')
#analyse dataset
# print(data.head())
# print(data.describe())
# print(data['species'].nunique())

#one vs all classification function
def classification(name,x):
    if x==name:
        return 1
    else:
        return 0

#add columns corresponding to one vs all
data['Iris-setosa'] = [classification('Iris-setosa',x) for x in data['species']]
data['Iris-versicolor'] = [classification('Iris-versicolor',x) for x in data['species']]
data['Iris-virginica'] = [classification('Iris-virginica',x) for x in data['species']]
#change each class to number
species = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
data['species'] = [species[x] for x in data['species']]

m = len(data)

#get the data x
x = np.array(data.iloc[:,:-4])
x = np.concatenate((np.ones((m,1)),x),axis=1)
#get the data y
y = np.array(data['species']).reshape((m,1))
#get all one vs all predictions
y1 = np.array(data['Iris-setosa']).reshape((m,1))
y2 = np.array(data['Iris-versicolor']).reshape((m,1))
y3 = np.array(data['Iris-virginica']).reshape((m,1))
# print(x)
# print(y)
# print(y1)
# print(y2)
# print(y3)


#train test split
def train_test_split(x,y,s):
    val = int(m*(1-s))

    indices = np.arange(m)
    np.random.shuffle(indices)

    A = x[indices]
    B = y[indices]
    B1 = y1[indices]
    B2 = y2[indices]
    B3 = y3[indices]

    x_train = A[:val+1]
    x_test = A[val+1:]
    y1_train = B1[:val+1]
    y2_train = B2[:val+1]
    y3_train = B3[:val+1]
    y_test = B[val+1:]

    return (x_train,y1_train,y2_train,y3_train,x_test,y_test)

#split data into train test and split
(x_train,y1_train,y2_train,y3_train,x_test,y_test) = train_test_split(x,y,0.3)
m = x_train.shape[0]

#sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

#hypothesis function
def hypothesis_function(theta):
    return sigmoid(x_train@theta)

#cost function vectorized implementation
def cost_function(theta,y_train):
    h_theta = hypothesis_function(theta)
    return (-1/m)*((y_train.T@np.log(h_theta)) + ((1-y_train).T@np.log(1-h_theta)))

#gradient descent algorithm vectorized implementation
def gradient_descent(theta,y_train):
    prev_cost = cost_function(theta,y_train)
    for i in range(10000):
        cur_cost = cost_function(theta,y_train)
        theta = theta - (0.1/m)*x_train.T@(hypothesis_function(theta)-y_train)
        if cur_cost>prev_cost:
            print(f"{i} error")
            break
        prev_cost = cur_cost
    return theta

theta1 = np.ones((5,1))
theta2 = np.ones((5,1))
theta3 = np.ones((5,1))
#run gradient descent and get theta parameters for each one vs all
theta1 = gradient_descent(theta1,y1_train)
theta2 = gradient_descent(theta1,y2_train)
theta3 = gradient_descent(theta1,y3_train)
print(theta1)
print(theta2)
print(theta3)

#test data and its size
m_test = x_test.shape[0]
#predicted y on test data for each class
y1_pred = sigmoid(x_test@theta1)
y2_pred = sigmoid(x_test@theta2)
y3_pred = sigmoid(x_test@theta3)

#multiclass classification : choose the class corresponding to highest probability
y_pred = []
for i in range(m_test):
    if y1_pred[i] > y2_pred[i]:
        if y1_pred[i] > y3_pred[i]:
            y_pred.append(0)
        else:
            y_pred.append(2)
    else:
        if y2_pred[i] > y3_pred[i]:
            y_pred.append(1)
        else:
            y_pred.append(2)

#model analysis
count=0
for i in range(m_test):
    #count number of correct predictions
    if y_pred[i]==y_test[i]:
        count+=1
#print accurate prediction percentage
print(count*100/m_test)
