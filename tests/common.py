# -*- coding: utf-8 -*-
import logging
import sys
import uuid

import numpy as np
import keras as k
import tensorflow as tf

from iwlearn.base import BaseSample, BaseFeature
# from iwlearn.models import BaseKerasClassifierModel
import random as r


class TestOfferSample(BaseSample):
    __test__ = False

    def __init__(self, entityid):
        BaseSample.__init__(self, entityid)
        self.data['_id'] = str(uuid.uuid4())


class OneHotFeature(BaseFeature):
    def __init__(self):
        BaseFeature.__init__(self)
        self.output_shape = (100,)

    def get(self, sample):
        result = np.zeros(self.output_shape)
        result[r.randint(0, 99)] = 1
        return result


# class SimpleKerasModel(BaseKerasClassifierModel):
#     def __init__(self, name, features, sampletype):
#         BaseKerasClassifierModel.__init__(self, name, features, sampletype, 2, labelkey='TestModelLabel')
#
#
#     def _createkerasmodel(self):
#         self.kerasmodel = k.models.Sequential(
#             [
#                 k.layers.Dense(self.input_shape[0], activation='relu', input_shape=self.input_shape),
#                 k.layers.Dense(self.input_shape[0], activation='relu'),
#                 k.layers.Dense(2, activation='softmax')
#             ]
#         )
#         self.kerasmodel.compile(optimizer=tf.train.AdamOptimizer(0.0002),
#                       loss='categorical_crossentropy',
#                       metrics=[k.metrics.categorical_accuracy])
#

class SimpleSampleFeature(BaseFeature):
    def __init__(self, key, width=1):
        BaseFeature.__init__(self)
        self.output_shape = (width,)
        self.key = key
        self.name += "_" + key

    def get(self, sample):
        return sample[self.key]


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
