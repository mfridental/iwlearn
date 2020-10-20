# -*- coding: utf-8 -*-
import sys
import logging

from tutorial.common.configuration import COUCHBASE_CONNECTION_STRING

from tutorial.common import _DUMMY   # do not use dummy in real projects, instead:
                # from iwlearn.datasources import CouchBaseDataSource, SQLDataSource, ...

class WatchedRealEstatesDataSource(_DUMMY.MockCouchBaseDataSource):
    def __init__(self):
        _DUMMY.MockCouchBaseDataSource.__init__(self, COUCHBASE_CONNECTION_STRING)

    def makeimpl(self, userkey):
        user_profile = self.get_document(userkey)
        return user_profile['VisitedRealEstates']  # list of estate ids


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)