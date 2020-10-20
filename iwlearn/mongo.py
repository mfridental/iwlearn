# -*- coding: utf-8 -*-
import sys
import logging
import datetime as dt
import numpy as np
import copy
import threading

from pymongo import MongoClient, DESCENDING, results
from bson import Binary
from decimal import Decimal
from pytz import timezone

from iwlearn.base import BasePrediction

MONGO_URI = None


def setmongouri(uri):
    global MONGO_URI
    MONGO_URI = uri

_mongoclient = {}
_mongoclientlock = threading.Lock()
def mongoclient():
    global _mongoclient, _mongoclientlock
    if MONGO_URI is None:
        raise Exception('Please set MONGO_URI after importing iwlearn')

    id = threading.get_ident()
    if id not in _mongoclient:
        _mongoclientlock.acquire()
        if id not in _mongoclient:
            _mongoclient[id] = MongoClient(MONGO_URI, maxPoolSize=1)
        _mongoclientlock.release()

    return _mongoclient[id]

def _convert_document_for_python(document):
    if isinstance(document, dict):
        result = dict()
        for key, value in document.items():
            result[str(key).replace('.', '_')] = _convert_document_for_python(value)
        return result
    if isinstance(document, list):
        return [_convert_document_for_python(x) for x in document]
    if isinstance(document, float):
        return Decimal(document)
    if isinstance(document, dt.datetime):
        return timezone('UTC').localize(document).astimezone(timezone('Europe/Berlin')).replace(tzinfo=None)
    if isinstance(document, dt.date):
        tmp_datetime = dt.datetime(document.year, document.month, document.day)
        utc_datetime = timezone('UTC').localize(tmp_datetime).astimezone(timezone('Europe/Berlin')).replace(tzinfo=None)
        return dt.date(utc_datetime.year, utc_datetime.month, utc_datetime.day)
    return document


def _convert_documents_for_python(documents):
    return [_convert_document_for_python(document) for document in documents]


def localize_datetime(data):
    return timezone('Europe/Berlin').localize(data).astimezone(timezone('UTC'))


def localize_date(data):
    tmp = dt.datetime.combine(data, dt.datetime.min.time())
    return localize_datetime(tmp)


def _convert_data_for_mongo(data):
    if isinstance(data, dict):
        result = dict()
        for key, value in data.items():
            result[str(key).replace('.', '_')] = _convert_data_for_mongo(value)
        return result
    if isinstance(data, np.ndarray):
        return _convert_data_for_mongo(data.tolist())
    if isinstance(data, np.float32):
        return float(data)
    if isinstance(data, list) or isinstance(data, tuple):
        return [_convert_data_for_mongo(x) for x in data]
    if isinstance(data, Decimal):
        return float(data)
    if isinstance(data, dt.datetime):
        return localize_datetime(data)
    if isinstance(data, dt.date):
        return localize_date(data)
    return data


def collectionname(sampletype):
    return sampletype.__name__ + 's'


def find_latest_sample(sampletype, entityid):
    if MONGO_URI is not None:
        coll = mongoclient()['IWLearn'][collectionname(sampletype)]
        docs = list(coll.find({'entityid': entityid}).sort('created', DESCENDING).limit(1))
        if len(docs) == 1:
            return sampletype.fromjson(_convert_document_for_python(docs[0]))
    logging.warning('Caching for %s, %s not available' % (sampletype.__name__, entityid))
    return None


def find_samples(sampletype, *args, **kwargs):
    coll = mongoclient()['IWLearn'][collectionname(sampletype)]
    if 'projections' in kwargs and 'entityid' not in kwargs['projection']:
        raise Exception('Include entityid into projection')
    samples = [sampletype.fromjson(_convert_document_for_python(document)) for document in coll.find(*args, **kwargs)]
    return samples


def find_samples_generator(sampletype, batch_size=1000, *args, **kwargs):
    coll = mongoclient()['IWLearn'][collectionname(sampletype)]
    if 'projections' in kwargs and 'entityid' not in kwargs['projection']:
        raise Exception('Include entityid into projection')
    cursor = coll.find(*args, **kwargs)
    samples = []
    batch_number = 0
    for doc in cursor:
        samples.append(sampletype.fromjson(_convert_document_for_python(doc)))
        if len(samples) == batch_size:
            logging.info('start %s batch with %s samples' % (batch_number, len(samples)))
            for sample in samples:
                yield sample
            samples = []
            batch_number += 1

    logging.info('start %s batch with %s samples' % (batch_number, len(samples)))
    for sample in samples:
        yield sample

def insert_sample(sample):
    prepared_doc = _convert_data_for_mongo(sample.data)

    client = mongoclient()
    coll = client['IWLearn'][collectionname(sample.__class__)]
    if '_id' not in sample.data:
        insert_data = coll.insert_one(prepared_doc)
        if isinstance(insert_data, results.InsertOneResult) and insert_data.acknowledged:
            sample.data['_id'] = insert_data.inserted_id
            logging.info('%r persisted' % sample)
        else:
            raise Exception('%r could not be inserted' % sample)
    else:
        raise Exception('Sample %r cannot be inserted, because it already has _id' % sample)

def replace_sample(sample):
    prepared_doc = _convert_data_for_mongo(sample.data)
    coll = mongoclient()['IWLearn'][collectionname(sample.__class__)]
    if '_id' not in sample.data:
        raise Exception('Sample %s cannot be replaced, because it doesn not have _id' % sample)
    else:
        replace_data = coll.replace_one({'_id': sample.id}, prepared_doc, upsert=True)
        if isinstance(replace_data, results.UpdateResult) and replace_data.acknowledged:
            pass
        else:
            raise Exception('%s could not be replaced' % sample)


def update_samples(samples, updatedoc):
    if len(samples) == 0:
        return
    prepared_doc = _convert_data_for_mongo(updatedoc)
    for collname in set([collectionname(x.__class__) for x in samples]):
        coll = mongoclient()['IWLearn'][collname]
        filter = {'_id': {'$in': [x.id for x in samples if collectionname(x.__class__) == collname]}}
        result = coll.update_many(filter=filter, update=prepared_doc)
        if not result.acknowledged or result.matched_count != len(samples):
            raise Exception('Cannot update (all) samples, %d' % result.matched_count)


def insert_samples(samples):
    prepared_docs = [_convert_data_for_mongo(sample.data) for sample in samples]

    if any([sample.__class__ != samples[0].__class__ for sample in samples]):
        raise Exception('Not all samples are objects of class %s' % samples[0].__class__)

    coll = mongoclient()['IWLearn'][collectionname(samples[0].__class__)]

    if any(['_id' in sample.data for sample in samples]):
        raise Exception("At least one sample already has an '_id' - batch replace is not supported")

    insert_data = coll.insert_many(prepared_docs, ordered=True)
    if isinstance(insert_data, results.InsertManyResult) and insert_data.acknowledged:
        for sample, inserted_id in zip(samples, insert_data.inserted_ids):
            sample.data['_id'] = inserted_id
    else:
        raise Exception('Samples could not be saved')


class MongoUriError(Exception):
    def __init__(self, message):
        super(MongoUriError, self).__init__(message)


def insert_check(prediction, sample, inserted=None):
    if inserted is None:
        inserted = []

    if prediction is not None and isinstance(prediction, BasePrediction):
        if prediction in inserted:
            return
        inserted.append(prediction)
        [insert_check(child, sample, inserted) for name, child in prediction.children.items()]

        data = copy.deepcopy(vars(prediction))

        if 'children' in data:
            data['children'] = {name: child for (name, child) in data['children'].items() if
                                not isinstance(child, BasePrediction)}

            if len(data['children'].keys()) == 0:
                data.pop('children')

        data['sample_id'] = sample.id
        data['created'] = sample.created
        data['entityid'] = sample.entityid

        prepared_doc = _convert_data_for_mongo(data)
        logging.debug(prepared_doc)

        collection_name = sample.name.replace('Sample', 'Check') + 's'
        coll = mongoclient()['IWLearn'][collection_name]
        logging.debug(coll)

        insert_data = coll.insert_one(prepared_doc)
        if isinstance(insert_data, results.InsertOneResult) and insert_data.acknowledged:
            inserted_id = insert_data.inserted_id
            if 'predictor' in data:
                logging.info('persisted %s %s %s' % (collection_name, data['predictor'], inserted_id))
            else:
                logging.info('persisted %s %s' % (collection_name, inserted_id))
        else:
            raise Exception('Prediction %d %s could not be persisted' % (sample, prepared_doc))


def update_checks_of_samples(samples, updatedoc):
    if len(samples) == 0:
        return
    prepared_doc = _convert_data_for_mongo(updatedoc)
    for collname in set([x.name.replace('Sample', 'Check') + 's' for x in samples]):
        coll = mongoclient()['IWLearn'][collname]
        filter = {'sample_id': {'$in': [x.id for x in samples if x.name.replace('Sample', 'Check') + 's' == collname]}}
        result = coll.update_many(filter=filter, update=prepared_doc)
        if not result.acknowledged:
            raise Exception('Cannot update checks')


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
