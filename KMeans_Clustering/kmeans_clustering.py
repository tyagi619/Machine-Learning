#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs

# read data
data = pd.read_csv('s1.txt',header=None,sep="   ",engine='python')
#data analysis
# print(data.head())
# print(data[0])
# print(data.describe())
# print(data.info())

#plot the data to get idea of number of clusters
# plt.scatter(data[0],data[1])
# plt.show()

# get data as numpy array
x = np.array(data)
m = x.shape[0]
# set number of clusters
k = 15

#train the model
def train(k_centres,c):
    sum_points = [] # sum of points assigned to each cluster
    number_points = [] # number of points assigned to each cluster
    # initialise both sum_points and number_points to zero
    for i in range(k):
        sum_points.append(np.zeros((x.shape[1],)))
        number_points.append(0)
    #number of iterations for model to run
    for i in range(100):
        # find the cluster centre nearest to each data point
        for j in range(m):
            dist = np.linalg.norm(x[j]-k_centres[0])
            c[j] = 0
            for l in range(1,k):
                cur_dist = np.linalg.norm(x[j]-k_centres[l])
                # update distance and cluster number if the distace to current cluster
                # is smaller than the min distance previously encountered
                if cur_dist<dist:
                    dist = cur_dist
                    c[j] = l  # update cluster number of the datapoint
            # update number_points and sum_points
            number_points[c[j]]+=1
            sum_points[c[j]]+=x[j]
        for j in range(k):
            if number_points[j]!=0:
                #update cluster centre as the mean of datapoints assigned to the cluster
                k_centres[j] = sum_points[j]/number_points[j]
                #set number_points and sum_points to zero for next iteration
                number_points[j] = 0;
                sum_points[j] = np.zeros((x.shape[1],))
    # return the final cluster centres and cluster number assigned to each datapoint
    return k_centres,c

k_centres = []  # stores the centre of each cluster
c = []  # stores the cluster assigned to each datapoint
#random initialization of cluster centres
random_points = np.random.randint(low=0,high=m,size=k)
for i in range(k):
    k_centres.append(x[random_points[i]])
for i in range(m):
    c.append(0)
#get the final cluster centres and cluster number assigned to each datapoint
k_centres,c = train(k_centres,c)
# print(c)

#plot and show different clusters on graph
colors = ['yellow','blue','red','green','black','pink','maroon','navy','grey','brown','gold','wheat','crimson','violet','lawngreen']
plt.scatter(data[0],data[1], c=c, cmap =clrs.ListedColormap(colors))
plt.show()
