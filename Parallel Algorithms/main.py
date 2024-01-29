'''
Luke Unterman - CMSC502
Main program - handles running each of the different
parallel DBSCAN implementations.
Using Wang et al.'s DBSCAN function from local pip install.
Link to Wang et al. GitHub:
https://github.com/wangyiqiu/dbscan-python
'''

import pandas as pd
from dbscan import DBSCAN
from pathlib import Path
from preprocess import preprocess
from sequential_DBSCAN import sDBSCAN
from parallel_DBSCAN import pDBSCAN
from time import time


def main():
    # Getting normalized data file
    filename = 'twitchdata-update.csv'
    path = Path('d:/CS/Parallel/DBSCAN/')
    outname = 'cleaned-twitchdata.csv'
    outpath = path / outname

    # Preprocessing data file according to Mochurad et al.
    if not outpath.exists():
        print('Preprocessing file!')
        preprocess(path / filename, outname)
    
    # Get data
    data = pd.read_csv(outpath).to_numpy()
    eps = 0.1
    min_pts = 35
    cluster = pDBSCAN(data, eps, min_pts)
    start = time()

    # Start threads
    cluster.init(6, 4)
    end = time()
    labels = cluster.labels
    print(labels)
    print(end-start)


    # Wang et al. implementation
    start = time()
    real_labels, core_samples_mask = DBSCAN(X=data, eps=eps, min_samples=min_pts)
    end = time()
    print(real_labels)
    print(f'elapsed: {end - start}')


if __name__ == '__main__':
    main()
