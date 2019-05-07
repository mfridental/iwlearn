# -*- coding: utf-8 -*-
import logging
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


def cluster_dataset(dataset, n_clusters, index_array = None, features = None):
    if index_array is None:
        index_array = np.arange(0, len(dataset))
    X, _ = dataset.get_samples(index_array, features)

    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    #algo = MiniBatchKMeans(n_clusters=n_clusters)
    algo = KMeans(n_clusters=n_clusters)
    labels = algo.fit_predict(X)

    sil = silhouette_score(X, labels)
    return labels, sil, algo

def stratified_split(dataset, test_size=0.25, verbose=False):
    _, y = dataset.get_all_samples(features=[])

    klasses = set(y)
    train_idx = []
    test_idx = []
    if verbose:
        logging.info('stratified_split')
    for klass in klasses:
        indices = np.where(y == klass)[0]
        train, test = train_test_split(indices, test_size=test_size)
        train_idx.extend(train)
        test_idx.extend(test)

        if verbose:
            logging.info('#class0: %d' % len(indices))

    return train_idx, test_idx

def balance_dataset(dataset, n_clusters, factor=1.5):
    X, y = dataset.get_all_samples()
    klasses = set(y)
    klass_indexes = {}
    smallest_klass_size = None
    for klass in klasses:
        klass_indexes[klass] = np.where(y==klass)[0]
        size = len(klass_indexes[klass])
        if smallest_klass_size is None or size < smallest_klass_size:
            smallest_klass_size = size

    newIdx = []
    newsize = int(smallest_klass_size * factor)
    for klass in klasses:
        size = len(klass_indexes[klass])
        if size < smallest_klass_size * factor:
            logging.info('Copying class %d of %d samples' % (klass, size))
            newIdx.extend(klass_indexes[klass])
            continue

        logging.info('Reducing class %d from %d to %d samples' % (klass, size, newsize))

        clusterlabels, sil, algo = cluster_dataset(dataset, n_clusters, klass_indexes[klass])
        counts = [sum(clusterlabels==i) for i in xrange(0, n_clusters)]

        X0 = X[klass_indexes[klass]]
        for cluster_num in xrange(0, n_clusters):
            X_c = X0[clusterlabels==cluster_num]
            nn = NearestNeighbors()
            nn.fit(X_c)
            numsamples = int(counts[cluster_num] * 1.0 / len(X) * newsize)
            logging.debug('cluster %d, samples %d' % (cluster_num, numsamples))
            ind = nn.kneighbors([algo.cluster_centers_[cluster_num]], numsamples, return_distance=False)[0]
            newIdx.extend(klass_indexes[klass][ind])

    return newIdx



if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)