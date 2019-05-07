# -*- coding: utf-8 -*-
import sys
import logging

import iwlearn.mongo as mongo
from iwlearn.training import DataSet

from tutorial.common.models import RelocationModelPro

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    mongo.setmongouri('mongodb://localhost:27017/')

    DataSet.remove('train-pro')
    DataSet.generate('train-pro', RelocationModelPro(), numclasses=2, filter={'entityid': {'$regex': r'^user[0-9]*?[0-7]$'}})

    DataSet.remove('test-pro')
    DataSet.generate('test-pro', RelocationModelPro(), numclasses=2, filter={'entityid': {'$regex': r'^user[0-9]*?[8-9]$'}})