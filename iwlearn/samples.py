# -*- coding: utf-8 -*-

from base import BaseSample
from datasources import *


class SQLSample(BaseSample, SQLDataSource):
    def __init__(self, entityid, connection_string):
        BaseSample.__init__(self, entityid)
        SQLDataSource.__init__(self, connection_string)


class CouchBaseSample(BaseSample, CouchBaseDataSource):
    def __init__(self, entityid, connection_string):
        BaseSample.__init__(self, entityid)
        CouchBaseDataSource.__init__(self, connection_string)


class MongoDBSample(BaseSample, MongoDBDataSource):
    def __init__(self, entityid, connection_string, database, collection, socketTimeoutMS=20000):
        BaseSample.__init__(self, entityid)
        MongoDBDataSource.__init__(self, connection_string, database, collection, socketTimeoutMS)


class ClickhouseSample(BaseSample, ClickhouseDataSource):
    def __init__(self, entityid, connection_string, timeout=5):
        BaseSample.__init__(self, entityid)
        ClickhouseDataSource.__init__(self, connection_string, timeout)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
