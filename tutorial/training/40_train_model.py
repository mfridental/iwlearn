# -*- coding: utf-8 -*-
import sys
import logging

from iwlearn.training import DataSet

from tutorial.common.models import RelocationModel

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train = DataSet('train')
    test = DataSet('test')
    print 'Samples in train %d, in test %d' % (len(train), len(test))

    model = RelocationModel()
    model.train(train)
    model.evaluate_and_print(test)
