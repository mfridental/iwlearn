# -*- coding: utf-8 -*-
import logging
import datetime as dt
import numpy as np
from decimal import Decimal
import copy
import re
from iwlearn.base import BasePrediction

class BaseStore(object):
    def __init__(self, sampletype):
        self.sampletype = sampletype

    @staticmethod
    def _convert_data_for_json(data):
        if isinstance(data, np.ndarray):
            return BaseStore._convert_data_for_json(data.tolist())
        if isinstance(data, np.float32):
            return float(data)
        if isinstance(data, list) or isinstance(data, tuple):
            return [BaseStore._convert_data_for_json(x) for x in data]
        if isinstance(data, dict):
            return dict([(str(k), BaseStore._convert_data_for_json(v)) for k, v in data.items()])
        if isinstance(data, Decimal):
            return float(data)
        if isinstance(data, dt.datetime):
            return data.strftime('%Y-%m-%dT%H:%M:%S.%f')
        if isinstance(data, dt.date):
            return data.strftime('%Y-%m-%d')
        return data

    @staticmethod
    def _convert_data_from_json(data):
        if isinstance(data, list) or isinstance(data, tuple):
            return [BaseStore._convert_data_from_json(x) for x in data]
        if isinstance(data, dict):
            return dict([(str(k), BaseStore._convert_data_from_json(v)) for k, v in data.items()])
        if isinstance(data, str) and re.match('^[0-9]{4}-[0-1][0-9]-[0-3][0-9]$', data) is not None:
            try:
                return dt.datetime.strptime(data, '%Y-%m-%d').date()
            except:
                return data
        if isinstance(data, str) and \
            re.match('^[0-9]{4}-[0-1][0-9]-[0-3][0-9]T[0-2][0-9]:[0-5][0-9]:[0-5][0-9]\.[0-9]{6}'
                                              '$', data) is not None:
            try:
                return dt.datetime.strptime(data, '%Y-%m-%dT%H:%M:%S.%f')
            except:
                return data
        return data


    def find_latest_sample(self, entityid):
        raise NotImplementedError

    def find_samples(self, *args, **kwargs):
        raise NotImplementedError

    def find_samples_generator(self, batch_size=1000, *args, **kwargs):
        raise NotImplementedError

    def insert_sample(self, sample):
        raise NotImplementedError

    def replace_sample(self, sample):
        raise NotImplementedError

    def insert_samples(self, samples):
        raise NotImplementedError

    def _insert_prediction(self, data):
        raise NotImplementedError

    def insert_prediction(self, prediction, sample, inserted=None):
        if inserted is None:
            inserted = []

        if prediction is not None and isinstance(prediction, BasePrediction):
            if prediction in inserted:
                return
            inserted.append(prediction)
            [self.insert_prediction(child, sample, inserted) for name, child in prediction.children.items()]

            data = copy.deepcopy(vars(prediction))

            if 'children' in data:
                data['children'] = {name: child for (name, child) in data['children'].items() if
                                    not isinstance(child, BasePrediction)}

                if len(data['children'].keys()) == 0:
                    data.pop('children')

            data['sample_id'] = sample.id
            data['created'] = sample.created
            data['entityid'] = sample.entityid

            self._insert_prediction(data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s\t%(levelname)s\t%(message)s')
