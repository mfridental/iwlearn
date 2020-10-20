# -*- coding: utf-8 -*-
import sys
import logging
import datetime as dt

from iwlearn import BaseRule, RulePrediction

import tutorial.common.features as ff
from tutorial.common.samples import RelocationUserSample

class RelocationRule(BaseRule):
    def __init__(self):
        BaseRule.__init__(self,
                          [
                              ff.LivingAreaMedian(),
                              ff.RoomMedian(),
                              ff.PercentageHousesForRent(),
                              ff.ExpensivePrice()],
                          sampletype=RelocationUserSample,
                          labelkey='RelocationLabel')

    def _implement_rule(self, livingAreaMedian, roomMedian, percentageHousesForRent, expensivePrice):
        if livingAreaMedian >= 75 and roomMedian >= 3 and \
            percentageHousesForRent > 0.3 and expensivePrice > 1.0:
            return RulePrediction(1)    # class 1 means "positive"
        return RulePrediction(0)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
