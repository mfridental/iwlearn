# -*- coding: utf-8 -*-
import logging
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from iwlearn.training.dataset import DataSet
from iwlearn.base import print_metric_results

def cluster_dataset(dataset, n_clusters, index_array=None, features=None):
    if index_array is None:
        index_array = np.arange(0, len(dataset))
    X, _ = dataset.get_samples(index_array, features)

    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    # algo = MiniBatchKMeans(n_clusters=n_clusters)
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
            logging.info('class %d: %d' % (klass, len(indices)))

    return train_idx, test_idx

def crossvalidated_evaluation(dataset, model_class, fold=10, test_size=0.25, train_class_sizes=None,
                              regenerate=False, **configuration):
    test_metrics = dict()
    train_metrics = dict()
    if not regenerate:
        for fnum in range(0, fold):
            train_name = '%s_train_%d' % (dataset.experiment_name, fnum)
            test_name = '%s_test_%d' % (dataset.experiment_name, fnum)
            if not DataSet.exists(train_name) or not DataSet.exists(test_name):
                regenerate = True
                break

    if regenerate:
        y = dataset.get_all_labels()
        numclasses = dataset.meta._estimate_num_classes()
        test_folds, train_folds = _generate_folds(y, fold, numclasses, train_class_sizes)

        for fnum in range(0, fold):
            train_name = '%s_train_%d' % (dataset.experiment_name, fnum)
            test_name = '%s_test_%d' % (dataset.experiment_name, fnum)
            DataSet.remove(train_name)
            DataSet.remove(test_name)
            dataset.clone(train_name, train_folds[fnum])
            dataset.clone(test_name, test_folds[fnum])

    for fnum in range(0, fold):
        logging.info('Fold [%d/%d]' % (fnum+1, fold))
        train_name = '%s_train_%d' % (dataset.experiment_name, fnum)
        test_name = '%s_test_%d' % (dataset.experiment_name, fnum)
        model = model_class()
        train = DataSet(train_name)
        model.train(train, **configuration)
        metrics, _,_ = model.evaluate(train)
        for k, v in metrics.items():
            if k not in train_metrics:
                train_metrics[k] = []
            train_metrics[k].append(v)
        test = DataSet(test_name)
        metrics, _,_ = model.evaluate(test)
        for k, v in metrics.items():
            if k not in test_metrics:
                test_metrics[k] = []
            test_metrics[k].append(v)
        logging.info('Fold training and test sets classes:')
        train.meta.print_label_summary()
        test.meta.print_label_summary()

    folded_train_metrics = dict()
    for k, v in train_metrics.items():
        vv = np.asarray(v)
        if np.issubdtype(vv.dtype, np.number):
            folded_train_metrics[k] = np.median(v, axis=0).tolist()
        else:
            folded_train_metrics[k] = v

    folded_test_metrics = dict()
    for k, v in test_metrics.items():
        vv = np.asarray(v)
        if np.issubdtype(vv.dtype, np.number):
            folded_test_metrics[k] = np.median(v, axis=0).tolist()
        else:
            folded_test_metrics[k] = v

    print ('Train metrics')
    print ('------------------------------------------------------')
    print_metric_results(folded_train_metrics)
    print()
    print()
    print ('Test metrics')
    print ('------------------------------------------------------')
    print_metric_results(folded_test_metrics)

    return train_metrics, test_metrics


def _generate_folds(y, fold, numclasses, train_class_sizes=None, test_class_sizes=None):
    y = np.asarray(y)
    klass_indexes = dict()
    minclass = None
    for klass in range(numclasses):
        tmp = np.asarray(np.where(y == klass)[0])
        np.random.shuffle(tmp)
        if minclass is None or minclass > len(tmp):
            minclass = len(tmp)
        klass_indexes[klass] = tmp

    if train_class_sizes is None:
        train_class_sizes = [1.0] * numclasses
    else:
        for klass in range(numclasses):
            train_class_sizes[klass] = train_class_sizes[klass] * minclass / len(klass_indexes[klass])

    if test_class_sizes is None:
        test_class_sizes = [1.0] * numclasses
    else:
        for klass in range(numclasses):
            test_class_sizes[klass] = test_class_sizes[klass] * minclass / len(klass_indexes[klass])


    train_folds = []
    test_folds = []
    for fnum in range(0, fold):
        trains = []
        tests = []
        for klass in range(numclasses):
            step = max(1, int(len(klass_indexes[klass])*test_class_sizes[klass] / fold))
            tests.append(klass_indexes[klass][fnum * step:(fnum + 1) * step])
            tmp = np.concatenate(
                (klass_indexes[klass][0:fnum * step], klass_indexes[klass][(fnum + 1) * step:]))
            np.random.shuffle(tmp)
            step = max(1,int(len(klass_indexes[klass])*train_class_sizes[klass] * (fold-1) / fold))
            trains.append(tmp[:step])
        train_folds.append(np.concatenate([x for x in trains if len(x) > 0]))
        test_folds.append(np.concatenate([x for x in tests if len(x) > 0]))
    return test_folds, train_folds


def balance_dataset(dataset, n_clusters, factor=1.5, target_class_sizes=None):
    if dataset.meta.label_shape != ():
        raise Exception('Balance dataset supports only datasets with the label shape of ()')

    klasses = range(0,dataset.meta._estimate_num_classes())

    X, y = dataset.get_all_samples()
    klass_indexes = dict()
    current_class_sizes = dict()
    for klass in klasses:
        klass_indexes[klass] = np.where(y == klass)[0]
        current_class_sizes[klass] = len(klass_indexes[klass])

    if target_class_sizes is None:
        target_class_sizes = dict()
        minsize = min(current_class_sizes.values())
        for k,v in current_class_sizes.items():
            target_class_sizes[k] = min(v, int(minsize*factor))

    newIdx = []
    for klass in klasses:
        if target_class_sizes[klass] >= current_class_sizes[klass]:
            logging.info('Copying class %d of %d samples' % (klass, current_class_sizes[klass]))
            newIdx.extend(klass_indexes[klass])
            continue

        logging.info('Reducing class %d from %d to %d samples' % (klass, current_class_sizes[klass],
                                                                  target_class_sizes[klass]))

        clusterlabels, sil, algo = cluster_dataset(dataset, n_clusters, klass_indexes[klass])
        counts = [sum(clusterlabels == i) for i in range(0, n_clusters)]

        X0 = X[klass_indexes[klass]]
        for cluster_num in range(0, n_clusters):
            X_c = X0[clusterlabels == cluster_num]
            nn = NearestNeighbors()
            nn.fit(X_c)
            numsamples = int(counts[cluster_num] * 1.0 / len(X) * target_class_sizes[klass])
            numsamples = max(numsamples, 1)
            logging.debug('cluster %d, samples %d' % (cluster_num, numsamples))
            ind = nn.kneighbors([algo.cluster_centers_[cluster_num]], numsamples, return_distance=False)[0]
            newIdx.extend(klass_indexes[klass][ind])

    return newIdx

def resize_class(dataset, klass, target_size, n_clusters=4):
    X, y = dataset.get_all_samples()
    klass_indexes = np.where(y == klass)[0]
    current_size = len(klass_indexes)

    if target_size == current_size:
        return klass_indexes

    if target_size > current_size:
        klass_indexes = np.concatenate((klass_indexes, klass_indexes[np.random.randint(len(klass_indexes)-1,
                                                                                       size=target_size-current_size)]))
        return klass_indexes

    clusterlabels, sil, algo = cluster_dataset(dataset, n_clusters, klass_indexes)
    counts = [sum(clusterlabels == i) for i in range(0, n_clusters)]

    newIdx = list()
    X0 = X[klass_indexes]
    for cluster_num in range(0, n_clusters):
        X_c = X0[clusterlabels == cluster_num]
        nn = NearestNeighbors()
        nn.fit(X_c)
        numsamples = int(counts[cluster_num] * 1.0 / len(X) * target_size)
        numsamples = max(numsamples, 1)
        logging.debug('cluster %d, samples %d' % (cluster_num, numsamples))
        ind = nn.kneighbors([algo.cluster_centers_[cluster_num]], numsamples, return_distance=False)[0]
        newIdx.extend(klass_indexes[ind])

    return newIdx


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
