# -*- coding: utf-8 -*-
import sys
import logging

import iwlearn.mongo as mongo
from iwlearn.training import DataSet

from tutorial.common.models import RelocationModelHyper

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    mongo.setmongouri('mongodb://localhost:27017/')

    DataSet.remove('train-hyper')
    DataSet.generate('train-hyper', RelocationModelHyper(), numclasses=2, filter={'entityid': {'$regex': r'^user[0-9]*?[0-7]$'}})

    DataSet.remove('test-hyper')
    DataSet.generate('test-hyper', RelocationModelHyper(), numclasses=2, filter={'entityid': {'$regex': r'^user[0-9]*?[8-9]$'}})