# -*- coding: utf-8 -*-
import sys
import logging
import datetime as dt

# DO NOT COPY THIS FILE IN REAL PROJECTS
# It contains only stuff needed for the tutorial

from iwlearn import BaseSample
from iwlearn.datasources import ThreadSafeDataSource
import random

MAGIC = 1000

class MockCouchBaseDataSource(ThreadSafeDataSource):
    def __init__(self, connstring):
        ThreadSafeDataSource.__init__(self)
        pass

    def get_document(self, key):
        if int(key[4:]) < MAGIC: # positive label for relocation. See also 01_create_dataset.py
            return {'VisitedRealEstates': [1,2,3]}
        else:
            return {'VisitedRealEstates': [4,5,6]}

class MockSQLSample(BaseSample):
    def __init__(self, entityid, connstring):
        BaseSample.__init__(self, entityid)
        pass

    def get_row_as_dict(self, query, *params):
        if params[0][0] in [1,3]: # positive label for relocation
            return {
                'price': random.gauss(1200,400),
                 'livingarea':random.gauss(120, 40),
                 'rooms':int(random.gauss(6,2)),
                 'zipcode':0,
                 'estatetype':'HOUSE' if random.random() > 0.2 else 'FLAT',
                 'distributiontype':'RENT'}

        else:
            return {
                'price': random.gauss(600,300),
                 'livingarea':random.gauss(60, 30),
                 'rooms':int(random.gauss(2,1)),
                 'zipcode':0,
                 'estatetype':'HOUSE' if random.random() > 0.8 else 'FLAT',
                 'distributiontype':'RENT'}


