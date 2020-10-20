# -*- coding: utf-8 -*-
import sys
import logging
import datetime as dt

from iwlearn.training import DataSet

from tutorial.common.models import RelocationModelPro

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train = DataSet('train-pro')
    test = DataSet('test-pro')
    print ('Samples in train %d, in test %d' % (len(train), len(test)))

    model = RelocationModelPro()
    model.train(train)
    model.evaluate_and_print(test)

    scored_features = model.feature_selection(test, step=1, n_splits=4)

    selected_features = []
    for i, (feature, score) in enumerate(zip(model.features, scored_features)):
        logging.info('%s %s %s ' % (i, feature.name, score))
        if score <= 1:
            selected_features.append(feature)
