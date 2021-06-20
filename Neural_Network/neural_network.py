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
m = x_train.shape[0]
#set alpha for gradient descent
alph = 1
# costs=[]
#sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

#hypothesis function
def hypothesis_function(theta):
    return sigmoid(x_train@theta.T)

#cost function vectorized implementation without regularization
def cost_function(theta):
    h_theta = hypothesis_function(theta)
    return (-1/m)*((y_train.T@np.log(h_theta)) + ((1-y_train).T@np.log(1-h_theta)))[0][0]

#gradient descent algorithm without regularization
def gradient_descent(theta):
    prev_cost = cost_function(theta)
    # costs.append(prev_cost)
    for i in range(1000):
        cap_delta = np.zeros((1,x_train.shape[1]))
        for j in range(m):
            #forward propagation
            a_j_0 = x_train[j].reshape((x_train.shape[1],1))
            a_j_1 = sigmoid(theta@a_j_0)
            #backward propagation
            delta = a_j_1 - y_train[j]
            #calculate delta
            cap_delta = cap_delta + delta@a_j_0.T
        #calculate derivative of cost function
        D = (1/m)*cap_delta
        #reduce theta in each iteration
        theta = theta - alph*D
        #check if gradient descent is running accurately
        cur_cost = cost_function(theta)
        # costs.append(cur_cost)
        if cur_cost>prev_cost:
            print(f"{i} error")
            break
        prev_cost = cur_cost
    return theta

#random initialisation of theta
theta = np.random.normal(size=x_train.shape[1]).reshape((1,x_train.shape[1]))
#run gradient descent and get theta parameters
theta = gradient_descent(theta)
print(theta)
#plot the cost function variation with each iteration
# plt.plot(costs)
# plt.show()

#test data and its size
m_test = x_test.shape[0]
#predicted y on test data
y_pred = sigmoid(x_test@theta.T)

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
