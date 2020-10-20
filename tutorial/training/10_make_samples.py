# -*- coding: utf-8 -*-
import sys
import logging

import iwlearn.mongo as mongo

from tutorial.common.samples import RelocationUserSample
from tutorial.common._DUMMY import MAGIC

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # for this tutorial, we use a local Mongo instance, which you can start by
    # docker run -p 27017:27017 -d mongo
    mongo.setmongouri('mongodb://10.200.0.1:27017/')
    mongo.mongoclient()['IWLearn']['RelocationUserSamples'].drop()

    # For this tutorial, we fake our data. In a real project,
    # load your real keys and labels directly from the databases
    DUMMY_userkeys = ['user' + str(x) for x in range(0, 10000)]
    DUMMY_userlabels = [1 if x < MAGIC else 0 for x in range(0, 10000)]

    for userkey, label in zip(DUMMY_userkeys, DUMMY_userlabels):
        sample = RelocationUserSample(userkey=userkey)
        sample['RelocationLabel'] = label
        sample.make()
        mongo.insert_sample(sample)
