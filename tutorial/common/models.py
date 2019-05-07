# -*- coding: utf-8 -*-
import sys
import logging
import datetime as dt

from iwlearn.models import ScikitLearnModel

import tutorial.common.features as ff
from tutorial.common.samples import RelocationUserSample

class RelocationModel(ScikitLearnModel):
    def __init__(self):
        ScikitLearnModel.__init__(self,
                                  'RelocationModel',
                                  features = [
                                    ff.LivingAreaMedian(),
                                    ff.RoomMedian(),
                                    ff.PercentageHousesForRent(),
                                    ff.ExpensivePrice()],
                                  sampletype=RelocationUserSample,
                                  labelkey='RelocationLabel')


class RelocationModelPro(ScikitLearnModel):
    def __init__(self):
        ScikitLearnModel.__init__(self,
                                  'RelocationModel',
                                  features = [
                                    ff.LivingAreaMedian(),
                                    ff.RoomMedianNoisy(),
                                    ff.PureNoise(),
                                    ff.PercentageHousesForRent(),
                                    ff.ExpensivePrice()],
                                  sampletype=RelocationUserSample,
                                  labelkey='RelocationLabel')

class RelocationModelHyper(ScikitLearnModel):
    def __init__(self):
        ScikitLearnModel.__init__(self,
                                  'RelocationModel',
                                  features = [
                                    ff.LivingAreaMedian(),
                                    ff.RoomMedianNoisy(),
                                    ff.PureStorm(),
                                    ff.PercentageHousesForRent(),
                                    ff.ExpensivePrice()],
                                  sampletype=RelocationUserSample,
                                  labelkey='RelocationLabel')



if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)