# -*- coding: utf-8 -*-
import pickle
import logging
import os
import sys
from shutil import copy

from bson import ObjectId

from iwlearn.mongo import mongoclient


def get(model_id):
    """
    :param model_id:
    :return: model
    """
    collection = mongoclient()['IWLearn']['Models']

    doc = collection.find_one(
        filter={'_id': ObjectId(model_id)},
        projection={'_id': 0, 'filepath': 1})

    with open(doc['filepath'], 'rb') as f:
        logging.info('load model from %s' % (doc['filepath']))
        return pickle.load(f)


def commit(model, creator):
    """
    :param creator:
    :param model:
    :return: filepath_modelname
    """
    dir = 'output/' + model.task
    if not os.path.isdir(dir):
        os.makedirs(dir)
    filepath_modelname = model._save(dir, creator)
    logging.info('filepath_modelname %s' % (filepath_modelname,))
    return filepath_modelname


def push(filepath_modelname):
    if filepath_modelname == '':
        raise Exception('appname is empty')

    copy(filepath_modelname, '/mnt/models/')


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
