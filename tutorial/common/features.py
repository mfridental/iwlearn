# -*- coding: utf-8 -*-
import sys
import logging
import datetime as dt
import numpy as np
import random

from iwlearn import BaseFeature

from tutorial.common.pricingengine import get_price

class LivingAreaMedian(BaseFeature):
    def __init__(self):
        BaseFeature.__init__(self)
        self.output_shape = ()

    def get(self, watchedRealEstateAttributes):
        if len(watchedRealEstateAttributes) == 0:
            return BaseFeature.MISSING_VALUE
        return np.median([estate['livingarea'] for estate in watchedRealEstateAttributes])

class RoomMedian(BaseFeature):
    def __init__(self):
        BaseFeature.__init__(self)
        self.output_shape = ()

    def get(self, watchedRealEstateAttributes):
        if len(watchedRealEstateAttributes) == 0:
            return BaseFeature.MISSING_VALUE
        return np.median([estate['rooms'] for estate in watchedRealEstateAttributes])

class PercentageHousesForRent(BaseFeature):
    def __init__(self):
        BaseFeature.__init__(self)
        self.output_shape = ()

    def get(self, watchedRealEstateAttributes):
        if len(watchedRealEstateAttributes) == 0:
            return BaseFeature.MISSING_VALUE
        return np.sum([1 for estate in watchedRealEstateAttributes
                       if estate['estatetype'] == 'HOUSE' and estate['distributiontype'] == 'RENT']) * 1.0 / \
               len(watchedRealEstateAttributes)

class ExpensivePrice(BaseFeature):
    def __init__(self):
        BaseFeature.__init__(self)
        self.output_shape = ()

    def get(self, watchedRealEstateAttributes):
        prices = []
        for estate in watchedRealEstateAttributes:
            if estate['distributiontype'] != 'RENT':
                continue
            hisprice = estate['price']
            meanprice = get_price(estate['zipcode'], estate['livingarea'], estate['estatetype'],
                                  estate['distributiontype'])
            prices.append((hisprice - meanprice) / meanprice)
        if len(prices) == 0:
            return BaseFeature.MISSING_VALUE

        return np.median(prices)


class RoomMedianNoisy(BaseFeature):
    def __init__(self):
        BaseFeature.__init__(self)
        self.output_shape = ()

    def get(self, watchedRealEstateAttributes):
        return np.median([estate['rooms'] for estate in watchedRealEstateAttributes]) + 100 * random.random()

class PureNoise(BaseFeature):
    def __init__(self):
        BaseFeature.__init__(self)
        self.output_shape = ()

    def get(self, watchedRealEstateAttributes):
        return random.random()

class PureStorm(BaseFeature):
    def __init__(self):
        BaseFeature.__init__(self)
        self.output_shape = (100, )

    def get(self, watchedRealEstateAttributes):
        return np.random.rand(self._get_width())


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)