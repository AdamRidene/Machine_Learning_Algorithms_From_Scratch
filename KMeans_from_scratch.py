import random
import numpy as np
x=[1,2,4,3,7,9]
y=[6,4,7,2,1,3]
data=list(zip(x,y))
class KMeans:
    def __init__(self,n_clusters,max_iter:int,n_init:int):
        self.n_clusters=n_clusters#number of clusters
        self.max_iter=max_iter#maximum number of iterations
        self.n_init=n_init#number of times to run the algorithm
        self.best_centroids=None
    def fit(self,data):
            self.data=data
            self.x=[i[0] for i in self.data]
            self.y=[i[1] for i in self.data]
            for _ in range(self.n_init):
                  self.centroids=[[random.choice(self.x),random.choice(self.y)] for i in range(self.n_clusters)]
                  self.costs=np.zeros(self.max_iter)
                  for it in range(self.max_iter):
                    #intialize clusters
                    self.clusters=[[] for _ in range(self.n_clusters)]
                    
                    for j in range(len(self.data)):
                         #calculate euclidean distance from each centroid
                         distances=[np.sqrt((self.x[j]-self.centroids[i][0]) **2 +(self.y[j]-self.centroids[i][1])**2) for i in range(self.n_clusters)]
                         #Assign data points to the nearest cluster
                         nearest_cluster=np.argmin(distances)
                         self.clusters[nearest_cluster].append(self.data[j])
                    for i in range(self.n_clusters):
                         if len(self.clusters[i])>0:
                             self.centroids[i]=[np.mean([j[0] for j in self.clusters[i]]),np.mean([j[1] for j in self.clusters[i]])]
                    #compute cost
                    cost=0
                    for i in range(len(self.data)):
                          distances=[np.sqrt((self.x[i]-self.centroids[j][0])**2 + (self.y[i]-self.centroids[j][1])**2) for j in range(self.n_clusters)]
                          cost+=np.mean((np.array(self.data[i])-np.array(self.centroids[np.argmin(distances)]))**2)
                    self.costs[it]=cost
                  self.best_centroids=self.centroids[np.argmin(self.costs)]
                         
    def cluster_centers(self):
            return self.best_centroids
    def predict(self,data):
          self.x=[i[0] for i in data]
          self.y=[i[1] for i in data]
          predictions=[]
          for i in range(len(data)):
                distances=[np.sqrt((self.x[i]-self.best_centroids[j][0])**2 + (self.y[i]-self.best_centroids[j][1])**2) for j in range(self.n_clusters)]
                predictions.append(np.argmin(distances))
          return predictions
          
