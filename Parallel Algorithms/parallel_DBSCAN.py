'''
Parallel (threaded) DBSCAN implementation
Translated C++ flow diagram of algorithm of Mochurad et al.
into Python
Note: some issues with race condition, output not always
exactly the same (slight difference)
'''

import numpy as np
from threading import Thread, Lock
from queue import Queue
from time import sleep


class pDBSCAN:
    def __init__(self, data, eps, min_pts):
        self.data = data
        self.neighbors = [Queue() for row in data]
        self.neighbors_found = np.zeros(shape=(len(data),), dtype=bool)
        self.eps = eps
        self.min_pts = min_pts
        self.lock = Lock()
        self.labels = np.zeros(shape=(len(data), ), dtype=int)

    # Euclidean n-dim distance
    def dist(self, A: int, B: int) -> float:
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)
            print('Converted')
        a_data = self.data[A]
        b_data = self.data[B]
        distance = np.sqrt(np.sum((a_data - b_data) ** 2))
        return distance
    
    # Getting neighbors, work divided by neighborWorkers
    # Lock is attempt at eliminating race condition
    def get_neighbors(self, start_value: int, step: int):
        n = len(self.data)
        for i in range(start_value - 1, n, step):
            with self.lock:
                is_found = self.neighbors_found[i]
            if not is_found:
                for j in range(i+1, n):
                    distance = self.dist(i, j)
                    if distance < self.eps:
                        self.neighbors[i].put(j)
                        self.neighbors[j].put(i)
                        
                with self.lock:
                    self.neighbors_found[i] = True


    # Non neighborWorkers wait for their point to be updated
    def wait_neighbors(self, point_index: int):
        while not self.neighbors_found[point_index]:
            sleep(0.01)
        return self.neighbors[point_index]

    # Once point received, start clustering
    # Basically just DBSCAN algorithm
    # (no real alterations)
    def cluster(self, start_value: int, step: int):
        n = len(self.data)
        current_cluster = 0
        for i in range(start_value - 1, n, step):
            if self.labels[i] == 0:
                neighbors_q = self.wait_neighbors(i)
                neighbors_q = list(neighbors_q.queue)
                if len(neighbors_q) < self.min_pts:
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
                        new_q = self.wait_neighbors(point)
                        new_q = list(new_q.queue)
                        if len(new_q) >= self.min_pts:
                            neighbors_q += new_q

    # Directs Thread workers based on wtype
    def control(self, index: int, start_value: int, step: int, wtype: str):
        if wtype == 'distance':
            self.get_neighbors(start_value, step)
        elif wtype == 'cluster':
            self.cluster(start_value, step)

    # Initializes threading
    # Threads access different subsequences via unique index + step for loops
    def init(self, neighbor_threads: int, cluster_threads: int):
        total_threads = neighbor_threads + cluster_threads
        threads = []
        worker_types = ['distance', 'cluster']
        for i in range(total_threads):
            thread_index = i + 1
            if i <= neighbor_threads:
                wtype = worker_types[0]  # Distance workers
                start_value = thread_index
                step = neighbor_threads
            else:
                wtype = worker_types[1]  # Cluster workers
                start_value = thread_index - neighbor_threads - 1
                step = cluster_threads - 1 
            t = Thread(target=self.control, args=(thread_index, start_value, step, wtype))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()