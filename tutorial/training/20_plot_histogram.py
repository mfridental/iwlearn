# -*- coding: utf-8 -*-
import logging
import sys

from iwlearn.training import DataSet

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train = DataSet('train')
    print 'Number of samples %d' % len(train)

    train.plot_data(bins=20)
