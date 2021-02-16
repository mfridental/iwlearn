# -*- coding: utf-8 -*-
import sys
import logging

import iwlearn.mongo as mongo
from iwlearn.training import DataSet

from tutorial.common.samples import RelocationUserSample
from tutorial.common.rules import RelocationRule

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    mongo.setmongouri('mongodb://localhost:27017/')

    DataSet.remove('train')
    DataSet.generate('train', RelocationRule(), filter={'entityid': {'$regex': r'^user[0-9]*?[0-7]$'}})

    DataSet.remove('test')
    DataSet.generate('test', RelocationRule(), filter={'entityid': {'$regex': r'^user[0-9]*?[8-9]$'}})