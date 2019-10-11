# The code for neural network has only 2 layers and no hidden layers
# This is the most basic implementation of neural network
# Some thing have been hard-coded keeping the above assumption in mind
# Kindly implement each part (forward propagation, back propagation) in separate functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read data
data = pd.read_csv('advertising.csv')
#analyse data
# print(data.head())
# print(data.describe())
m = len(data)
#get the x and y of data
x = np.array(data.drop(['Ad Topic Line','City','Country','Timestamp','Clicked on Ad'],axis=1))
y = np.array(data['Clicked on Ad']).reshape((m,1))

#mean normalization of data (except gender column)
for i in range(x.shape[1]-1):
    mean_x = np.mean(x[:][i])
    sd_x = np.std(x[:][i])
    for j in range(m):
        x[j][i] = (x[j][i]-mean_x)/sd_x
#add x0=1 for each data row
x = np.concatenate((np.ones((m,1)),x),axis=1)

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

#split the data into training and testing
(x_train,y_train,x_test,y_test) = train_test_split(x,y,0.3)
#training data size
m = x_train.shape[0]
#set alpha for gradient descent
alph = 0.1
#set number of layers
l = 4
#set number of nodes in each layer
hid = [x_train.shape[1]-1,4,4,1]

costs=[]

#sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

# forward propagation algorithm
def forward_propagation(theta,j):
    a = []
    a.append(x_train[j].reshape(x_train.shape[1],1))
    a_i = a[0]
    for i in range(l-1):
        z_i = theta[i]@a_i
        a_i = sigmoid(z_i)
        if i!=l-2:
            #add bias node to each layer except input and output layer
            a_i = np.concatenate((np.ones((1,1)),a_i),axis=0)
        a.append(a_i)
    return a

#back propagation algorithm - calculates delta for each layer and return list
def backward_propagation(theta,a,j):
    delta = []
    delta.append(a[-1]-y_train[j])
    delta_i = delta[0]
    for i in range(l-2):
        delta_i = (theta[l-i-2].T@delta_i)*(a[l-2-i])*(1-a[l-2-i])
        #back propagation is implemented excluding bias layer in each hidden layer
        delta_i = delta_i[1:]
        delta.append(delta_i)
    delta = delta[::-1]
    return delta

#calculates error of each layer
def calculate_layer_error(delta,a):
    cap_delta = []
    for i in range(l-1):
        cap_delta.append(delta[i]@a[i].T)
    return cap_delta

#cost function vectorized implementation without regularization
def cost_function(theta):
    h_theta = []
    for i in range(m):
        a = forward_propagation(theta,i)
        h_theta.append(a[-1][0][0])
    h_theta = np.array(h_theta).reshape((m,1))
    return (-1/m)*((y_train.T@np.log(h_theta)) + ((1-y_train).T@np.log(1-h_theta)))[0][0]

#gradient descent algorithm without regularization
def gradient_descent(theta):
    prev_cost = cost_function(theta)
    costs.append(prev_cost)
    #number of iterations of gradient descent
    for i in range(5000):
        #capital delta to calculate derivative of cost function
        cap_delta = []
        for j in range(l-1):
            cap_delta.append(np.zeros_like(theta[j]))
        #for each dataset
        for j in range(m):
            #forward propagation to calculate a(l)
            a = forward_propagation(theta,j)
            #backward propagation to calculate delta(l)
            delta = backward_propagation(theta,a,j)
            #calculate error of each layer
            cap_d = calculate_layer_error(delta,a)
            #sum up the error in each iteration for all datapoints
            for k in range(l-1):
                cap_delta[k]+=cap_d[k]
        # derivative of cost function
        D = []
        for j in range(l-1):
            D.append((1/m)*cap_delta[j])
        #update theta list
        for j in range(l-1):
            theta[j]-=alph*D[j]
        #calculate cost function for current parameters
        cur_cost = cost_function(theta)
        costs.append(cur_cost)
        #check if current cost is decreasing and not increasing with each iteration
        if(cur_cost>prev_cost):
            print(f"{i} error")
            break
        prev_cost = cur_cost
    return theta

#random initialisation of theta
theta_array = []
# create l-1 theta vector matrices for each layer
for i in range(l-1):
    theta = np.random.randn(hid[i+1],hid[i]+1)
    theta_array.append(theta)
#run gradient descent and get theta parameters
theta_array = gradient_descent(theta_array)
#print the theta parameters
for i in range(l-1):
    print(theta_array[i])
#plot the cost function variation with each iteration
plt.plot(costs)
plt.show()

#predict on test dataset
def predict(theta,j):
    a = []
    a.append(x_test[j].reshape(x_test.shape[1],1))
    a_i = a[0]
    for i in range(l-1):
        z_i = theta[i]@a_i
        a_i = sigmoid(z_i)
        if i!=l-2:
            a_i = np.concatenate((np.ones((1,1)),a_i),axis=0)
        a.append(a_i)
    return a[l-1][0]

#test data and its size
m_test = x_test.shape[0]
#predicted y on test data
y_pred = []
for i in range(m_test):
    y_pred.append(predict(theta_array,i))

#analysing the accuracy of model
tp=0 #true positive
tn=0 #true negative
fp=0 #false positive
fn=0 #false negative
for i in range(m_test):
    if y_pred[i]>=0.5:
        if y_test[i]==1:
            tp+=1
        else:
            fp+=1
    else:
        if y_test[i]==1:
            fn+=1
        else:
            tn+=1
#calculate recall,precision and F-score
recall = tp/(tp+fn)
precision = tp/(tp+fp)
f1_score = 2*precision*recall/(precision+recall)
#print the recall, precision and F-score
print(f"Precision = {precision}")
print(f"Recall = {recall}")
print(f"F1 Score = {f1_score}")
