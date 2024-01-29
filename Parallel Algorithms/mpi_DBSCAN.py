'''
My own parallel DBSCAN algorithm that uses MPI
Somewhat different from pDBSCAN:
Workers all evenly get neighbors,
then one slave gets labels for everyone else
Neighbors, Labels = communicated with bcast()
'''

from mpi4py import MPI
import numpy as np
import pandas as pd
from time import time

class mDBSCAN:
    def __init__(self, data, eps, min_pts):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.data = data
        self.neighbors = {}
        self.neighbors_number = []
        self.neighbors_found = np.zeros(shape=(len(data),), dtype=bool)
        self.eps = eps
        self.min_pts = min_pts
        self.labels = self.labels = {i:0 for i in range(len(self.data))}

    # euclidean n dimensional distance
    def dist(self, A: int, B: int) -> float:
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)
            print('Converted')
        a_data = self.data[A]
        b_data = self.data[B]
        distance = np.sqrt(np.sum((a_data - b_data) ** 2))
        return distance

    # Range Query
    def get_neighbors(self, start_value: int, step: int):
        n = len(self.data)
        for i in range(start_value - 1, n, step):
            is_found = self.neighbors_found[i]
            if not is_found:
                for j in range(i+1, n):
                    distance = self.dist(i, j)
                    if distance < self.eps:

                        # Different ways of handling the appending of neighbors
                        # in dictionary that is not preinitialized
                        if i not in self.neighbors and j not in self.neighbors:
                            self.neighbors[i] = [j]
                            self.neighbors[j] = [i]
                        elif i not in self.neighbors:
                            self.neighbors[i] = [j]
                            self.neighbors[j].append(i)
                        elif j not in self.neighbors:
                            self.neighbors[j] = [i]
                            self.neighbors[i].append(j)
                        else:
                            self.neighbors[i].append(j)
                            self.neighbors[j].append(i)

                    self.neighbors_found[i] = True

    # Base DBSCAN algorithm
    # Done sequentially
    def cluster(self, start_value: int, step: int):
        n = len(self.data)
        current_cluster = 0
        for i in range(start_value - 1, n, step):
            if self.labels[i] == 0:
                neighbors_q = self.neighbors[i]
                if self.neighbors_number[i] < self.min_pts:
                    self.labels[i] = -1
                    continue
                current_cluster += 1

                cluster_copy = current_cluster
                self.labels[i] = cluster_copy
                while len(neighbors_q) != 0:
                    point = neighbors_q.pop()
                    if self.labels[point] == -1:
                        self.labels[point] = cluster_copy
                    elif self.labels[point] == 0:
                        self.labels[point] = cluster_copy
                        new_q = self.neighbors[point]
                        if self.neighbors_number[point] >= self.min_pts:
                            neighbors_q += new_q

    # MPI control structure
    def control(self, start_value: int, step: int, wtype: str, neighbor_threads: int,
                 cluster_threads: int):
        
        # Each MPI processor calculates neighbors subsequences
        if wtype == 'distance':
            self.get_neighbors(start_value, step)
    
        # Join in list
        gather = self.comm.gather(self.neighbors, root=0)
        new_list = [[] for _ in range(len(self.data))]
        new_nf = np.zeros((len(self.data), ))

        # Stictch together lists of neighbors in proper order
        if self.rank == 0:
            for p in gather:
                for k, v in p.items():
                    new_list[k] = list(v)
                    new_nf[k] = len(new_list[k])
            #print(new_nf)
                    
        # Broadcast both lists from rank 0 
        self.neighbors = self.comm.bcast(new_list, root=0)
        self.neighbors_number = self.comm.bcast(new_nf, root=0)

        # Sequentially get labels
        new_labels = np.zeros((len(self.data), ))
        if self.rank == 0:
            self.cluster(1, 1)
            for k, v in self.labels.items():
                new_labels[k] = v

        # Now each other processing unit has labels
        self.labels = self.comm.bcast(new_labels, root=0)
        if self.rank == 0:
            print(self.labels)

    # Gives MPI processes starting, values, steps
    # Also each initially set as a neighborWorker
    def init(self, neighbor_threads: int, cluster_threads: int):
        total_threads = neighbor_threads + cluster_threads
        worker_types = ['distance', 'cluster']
        thread_index = self.rank + 1
        wtype = worker_types[0]  # Distance workers
        start_value = thread_index
        step = neighbor_threads
        self.control(start_value, step, wtype, neighbor_threads, cluster_threads)

data = pd.read_csv('cleaned-twitchdata.csv').to_numpy()
cluster = mDBSCAN(data, 0.1, 35)
start = time()
cluster.init(cluster.size, 0)
end = time()
if cluster.rank == 0:
    print(end-start)