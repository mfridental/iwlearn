# -*- coding: utf-8 -*-
import sys
import logging
import datetime as dt

import numpy as np

from iwlearn.training import DataSet

from tutorial.common.models import RelocationModel

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train = DataSet('train')
    test = DataSet('test')
    print ('Samples in train %d, in test %d' % (len(train), len(test)))

    model = RelocationModel()
    model.train(train)

    logging.info('Validation Curve n_estimators')
    n_estimators_range = np.linspace(start=30, stop=110, num=int((110 - 30) / 3), dtype=int)
    logging.info(len(n_estimators_range))
    plt = model.plot_validation_curve(dataset=train, param_name="n_estimators", param_range=n_estimators_range)
    plt.show()

    logging.info('Validation Curve max_features')
    max_features_range = np.linspace(start=1, stop=4, dtype=int)
    logging.info(len(max_features_range))
    plt = model.plot_validation_curve(dataset=train, param_name="max_features", param_range=max_features_range)
    plt.show()

    logging.info('Validation Curve min_samples_leaf')
    min_samples_leaf_range = np.linspace(start=1, stop=6, num=6, dtype=int)
    logging.info(len(min_samples_leaf_range))
    plt = model.plot_validation_curve(dataset=train, param_name="min_samples_leaf", param_range=min_samples_leaf_range)
    plt.show()

