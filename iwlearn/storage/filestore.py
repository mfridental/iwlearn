# -*- coding: utf-8 -*-
import logging
import datetime as dt
import re
import os
import ujson as json
import uuid

from scandir import scandir

from iwlearn.storage.base import BaseStore

class FileStore(BaseStore):
    def __init__(self, sampletype, basepath = 'input/filestore'):
        BaseStore.__init__(self, sampletype)
        self.samplebase = os.path.join(basepath, os.path.join('samples',sampletype.__name__))
        self.predictionbase = os.path.join(basepath, os.path.join('predictions',sampletype.__name__))

    def _scan(self, strategy):
        for entry in scandir(self.samplebase):
            if entry.is_dir():
                try:
                    day = dt.datetime.strptime(entry.name, '%Y-%m-%d')
                    if not strategy.is_part(day):
                        continue
                    for entry2 in scandir(entry.path):
                        if entry2.is_file():
                            try:
                                eid = entry2.name[0:-7]
                                if not strategy.is_entity(eid):
                                    continue
                                timeofday = dt.datetime.strptime(entry2.name[-6:], '%H%M%S')
                                day = day.replace(hour=timeofday.hour, minute=timeofday.minute,
                                            second=timeofday.second)
                                if strategy.is_fetch(day, eid):
                                    id = os.path.join(entry.name, entry2.name)
                                    strategy.add_fetch(id)
                            except:
                                logging.error('error fetching')
                except:
                    logging.error('error part selection')

    def find_latest_sample(self, entityid):
        class LatestSearch:
            def __init__(self, entityid):
                self.entityid = entityid
                self.latest = None
                self.result = None

            def is_part(self, day):
                return True

            def is_entity(self, eid):
                return eid == self.entityid

            def is_fetch(self, datetime, eid):
                if self.latest is None or self.latest < datetime:
                    self.latest = datetime
                    return True
                return False

            def add_fetch(self, id):
                self.result = id

        strategy = LatestSearch(entityid)
        self._scan(strategy)
        if strategy.result is None:
            return None
        return self._load(strategy.result)

    def _load(self, id):
        with open(os.path.join(self.samplebase, id), 'r') as f:
            doc = json.load(f)
            doc = BaseStore._convert_data_from_json(doc)
            sample = self.sampletype.fromjson(doc)
            sample['_id'] = id
            return sample

    def find_samples(self, *args, **kwargs):
        """
        Supports the following filter conditions:
            earliest : Date or DateTime
            latest = Date or DateTime
            match_entityid = Lambda getting entityid and returning boolean
        """
        return list(self.find_samples_generator(*args, **kwargs))

    def find_samples_generator(self, batch_size=1000, *args, **kwargs):
        """
        Supports the following filter conditions:
            earliest : Date or DateTime
            latest = Date or DateTime
            match_entityid = Lambda getting entityid and returning boolean
        """

        class FilterStrategy():
            def __init__(self, kwargs):
                self.filter = kwargs
                self.result = []
                if 'earliest' in self.filter:
                    if type(self.filter['earliest']) == dt.date:
                        self.filter['earliest'] = dt.datetime(
                            year=self.filter['earliest'].year,
                            month=self.filter['earliest'].month,
                            day=self.filter['earliest'].day,
                        )
                    if type(self.filter['earliest']) == dt.datetime:
                        self.filter['earliest'] = self.filter['earliest'].replace(microsecond=0)

                if 'latest' in self.filter:
                    if type(self.filter['latest']) == dt.date:
                        self.filter['latest'] = dt.datetime(
                            year=self.filter['latest'].year,
                            month=self.filter['latest'].month,
                            day=self.filter['latest'].day,
                            hour=23, minute=59, second=59
                        )
                    if type(self.filter['latest']) == dt.datetime:
                        self.filter['latest'] = self.filter['latest'].replace(microsecond=0)

            def is_part(self, day):
                if 'earliest' in self.filter:
                    if day.date() < self.filter['earliest'].date():
                        return False

                if 'latest' in self.filter:
                    if day.date() > self.filter['latest'].date():
                        return False

                return True

            def is_entity(self, eid):
                return 'match_entityid' not in self.filter or self.filter['match_entityid'](eid)

            def is_fetch(self, datetime, eid):
                if 'earliest' in self.filter:
                    if datetime < self.filter['earliest']:
                        return False

                if 'latest' in self.filter:
                    if datetime > self.filter['latest']:
                        return False

                return True

            def add_fetch(self, id):
                self.result.append(id)

        strategy = FilterStrategy(kwargs)
        self._scan(strategy)
        for id in strategy.result:
            yield self._load(id)


    def _get_paths(self, sample):
        part = sample.created.strftime('%Y-%m-%d')
        file = re.sub('[^a-zA-Z0-9_.\\-]', '_', sample.entityid) + \
               '_' + sample.created.strftime('%H%M%S')
        dir = os.path.join(self.samplebase, part)
        fullpath = os.path.join(dir, file)
        id = os.path.join(part, file)
        return dir, fullpath, id

    def insert_sample(self, sample):
        dir, fullpath, id = self._get_paths(sample)

        data = BaseStore._convert_data_for_json(sample.data)
        self._insert_data(dir, fullpath, data)
        sample['_id'] = id

    def replace_sample(self, sample):
        if '_id' in sample:
            fullpath = os.path.join(self.samplebase, sample['_id'])
        else:
            _, fullpath, _ = self._get_paths(sample)
        if os.path.isfile(fullpath):
            tmp = uuid.uuid4().hex
            os.rename(fullpath, fullpath + '__' + tmp)
            self.insert_sample(sample)
            os.remove(fullpath + '__' + tmp)

    def insert_samples(self, samples):
        for sample in samples:
            self.insert_sample(sample)

    def _insert_prediction(self, data):
        part = data['created'].strftime('%Y-%m-%d')
        file = re.sub('[^a-zA-Z0-9_.\\-]', '_', data['entityid']) + \
               '_' + data['created'].strftime('%H%M%S')
        dir = os.path.join(self.predictionbase, part)
        fullpath = os.path.join(dir, file)
        self._insert_data(dir, fullpath, data)

    def _insert_data(self, dir, fullpath, data):
        os.makedirs(dir, exist_ok=True)
        if os.path.isfile(fullpath):
            raise Exception('Cannot insert data under %s: its already exists' % fullpath)
        with open(fullpath, 'w') as f:
            json.dump(data, f)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(levelname)s\t%(message)s')
