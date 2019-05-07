# -*- coding: utf-8 -*-
import logging
import sys

from iwlearn.training import DataSet
from tutorial.common.rules import RelocationRule

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    test = DataSet('test')
    rule = RelocationRule()
    rule.evaluate_and_print(test)