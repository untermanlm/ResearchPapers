'''
Base DBSCAN algorithm
Using subsequences approach (just in loop) 
Serial comparison
'''

import numpy as np

class sDBSCAN:
    def __init__(self, data, eps, min_pts):
        self.data = data
        self.neighbors = [[] for row in data]
        self.neighbors_number = np.full(shape=(len(data), ), fill_value=-100, dtype=int)
        self.neighbors_found = np.zeros(shape=(len(data), ), dtype=bool)
        self.eps = eps
        self.min_pts = min_pts
        self.labels = np.zeros(shape=(len(data), ), dtype=int)

    # n dimensional euclidean distance
    def dist(self, A: int, B: int) -> float:
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)
            print('Converted')
        a_data = self.data[A]
        b_data = self.data[B]
        distance = np.sqrt(np.sum((a_data - b_data) ** 2))
        return distance
    
    # Range query 
    def get_neighbors(self, start_value: int, step: int):
        n = len(self.data)
        for i in range(start_value - 1, n, step):
            if not self.neighbors_found[i]:
                for j in range(i+1, n):
                    distance = self.dist(i, j)
                    if distance < self.eps:
                        self.neighbors[i].append(j)
                        self.neighbors[j].append(i)
                self.neighbors_number[i] = len(self.neighbors[i])
                self.neighbors_found[i] = True
        return self.neighbors, self.neighbors_number
    
    # Labels updated sequentially
    # Base DBSCAN algorithm
    def cluster(self, start_value: int, step: int):
        n = len(self.data)
        current_cluster = 0
        for i in range(start_value - 1, n, step):
            if self.labels[i] == 0:
                neighbors_i = self.neighbors[i]
                neighbors_number_i = self.neighbors_number[i]
                if neighbors_number_i < self.min_pts:
                    self.labels[i] = -1
                    continue
                current_cluster += 1
                
                cluster_copy = current_cluster
                self.labels[i] = cluster_copy

                # adding new members to cluster
                j = 0
                while j < len(neighbors_i):
                    seed = neighbors_i[j]
                    if self.labels[seed] == -1:
                        self.labels[seed] = cluster_copy
                    elif self.labels[seed] == 0:
                        self.labels[seed] = cluster_copy
                        new_neighbors = self.neighbors[seed]
                        new_neighbors_number = self.neighbors_number[seed]
                        if new_neighbors_number >= self.min_pts:
                            neighbors_i = neighbors_i + new_neighbors
                    j += 1    
        return self.labels

    # Initializes "workers"
    def init(self, neighbor_threads: int, cluster_threads: int):
        total_threads = neighbor_threads + cluster_threads
        for i in range(total_threads):
            thread_index = i + 1
            if i <= neighbor_threads:
                start_value = thread_index
                step = neighbor_threads
                self.get_neighbors(start_value, step)
            else:
                start_value = thread_index - neighbor_threads - 1
                step = cluster_threads - 1
                self.cluster(start_value, step)