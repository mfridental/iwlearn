# -*- coding: utf-8 -*-
import logging
import sys

import cherrypy

import iwlearn.mongo as mongo

from tutorial.common.rules import RelocationRule
from tutorial.common.samples import RelocationUserSample


class RelocationService():
    def __init__(self):
        self.rule = RelocationRule()
        self.collection = mongo.mongoclient()['Tutorial']['Predictions']

    @cherrypy.expose
    def predict(self, userkey):
        sample = RelocationUserSample(userkey=userkey)

        if sample.make() is not None:
            mongo.insert_sample(sample)

            prediction = self.rule.predict(sample)
            mongo.insert_check(prediction, sample)     # save prediction for rule monitoring and evaluation
            pre_dict = prediction.create_dictionary()  # this will serialize the prediction to a way suitable
                                                       # for storing in MongoDB or transmitting over the wire
            return pre_dict
        return 'cannot make sample'

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)



