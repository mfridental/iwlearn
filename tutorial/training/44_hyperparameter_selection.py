# -*- coding: utf-8 -*-
import sys
import logging
import datetime as dt

import numpy as np

from iwlearn.training import DataSet

from tutorial.common.models import RelocationModelHyper

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train = DataSet('train-hyper')
    test = DataSet('test-hyper')
    print 'Samples in train %d, in test %d' % (len(train), len(test))

    model = RelocationModelHyper()

    # Train the model in a simple way to provide a baseline of the model performance
    model.train(train)
    model.evaluate_and_print(test)

    print
    print

    # Now perform training with the hyperparameter optimization
    n_estimators_range = np.linspace(start=100, stop=600, num=5, dtype=int)
    max_features_range = np.linspace(start=5, stop=10, num=3, dtype=int)
    min_samples_leaf_range = np.linspace(start=1, stop=4, num=2, dtype=int)

    folds = 5 # use TrainingsSizeCrossValidation.xlsx and learningCurve to number of folds right
    param_distributions = {"n_estimators": n_estimators_range, "max_features": max_features_range, "min_samples_leaf": min_samples_leaf_range}

    configuration = {'n_splits': folds, 'param_distributions': param_distributions, 'params_distributions_test_proportion': 0.2, 'test_size': 0.25}
    model = RelocationModelHyper()
    model.train_with_hyperparameters_optimization(dataset=train, **configuration)
    model.evaluate_and_print(test)