# -*- coding: utf-8 -*-
import sys
import logging
import datetime as dt

from iwlearn import BaseSample


from tutorial.common.configuration import SQL_CONNECTION_STRING
from tutorial.common.datasources import WatchedRealEstatesDataSource
from tutorial.common import _DUMMY   # do not use dummy in real projects, instead:
                # from iwlearn.datasources import CouchBaseDataSource, SQLDataSource, ...


class RealEstateSample(_DUMMY.MockSQLSample):
    def __init__(self, estateid):
        _DUMMY.MockSQLSample.__init__(self, estateid, SQL_CONNECTION_STRING)

    def makeimpl(self):
        query = """
        select price, livingarea, rooms, zipcode, estatetype, distributiontype
        from RealEstates
        where estateid = ?
        """
        params = (self.entityid, )
        return self.get_row_as_dict(query, params)


class RelocationUserSample(BaseSample):
    def __init__(self, userkey):
        BaseSample.__init__(self, userkey) # this will set userkey to self.entityid

    def makeimpl(self):
        # The following line will make WatchedRealEstatesDataSource and add its result to the self under the key
        # with the same name as the name of the data source, without the "DataSource" suffix.

        # Because the data source needs to know, data of which user has to be retrieved, we can pass it here, because
        # all **kwargs parameter passed to self.add will be passed to the makeimpl method of the data source.

        # note that self.add also returns the retrieved dict(), so we can store it into estate_ids variable and use
        # later in our makeimpl code
        estate_ids = self.add(WatchedRealEstatesDataSource(), userkey=self.entityid)

        # now we just iterate the estateids and load the RealEstateSamples
        self['WatchedRealEstateAttributes'] = []
        for estateid in estate_ids:
            sub_sample = RealEstateSample(estateid)
            try:
                sub_sample.make()
                self.data['WatchedRealEstateAttributes'].append(sub_sample.data)
            except:
                logging.exception('Cannot make sample %s' % estateid)
        return self.data

        # result:
        # self.data['WatchedRealEstatesDataSource'] = [list of estateids]
        # self.data['WatchedRealEstateAttributes'] = [ {'price': .., 'livingarea': .., ..}, ...]


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)