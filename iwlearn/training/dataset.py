# -*- coding: utf-8 -*-
import pickle
import logging
import os
import re
import shutil
import sys
import traceback
import ujson as json
from bisect import bisect_right
from collections import Counter
from hashlib import sha256
from lru import LRU
from math import floor, ceil, sqrt, pow, log

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scandir import scandir
from tqdm import tqdm

from iwlearn.base import BaseFeature, BaseModel, Settings, create_tensor, combine_tensors, BaseSample
from iwlearn.mongo import mongoclient

class DataSetMeta(object):
    def __init__(self):
        self.features = []
        self.featurestats = dict()
        self.model_input_shape = None
        self.labels = []
        self.labelstats = dict()
        self.label_shape = None
        self.sampletype = None

    def copy(self, reset_stats=False):
        other = DataSetMeta()
        other.features = self.features.copy()
        other.featurestats = self.featurestats.copy()
        other.model_input_shape = self.model_input_shape
        other.labels = self.labels.copy()
        other.labelstats = self.labelstats.copy()
        other.label_shape = self.label_shape
        other.sampletype = self.sampletype

        if reset_stats:
            for k, v in other.featurestats.items():
                if 'top10values' in v:
                    v['top10values'] = [Counter() for _ in v['top10values']]
            for k, v in other.labelstats.items():
                if 'top10values' in v:
                    v['top10values'] = [Counter() for _ in v['top10values']]

        return other

    def getfeaturemap(self, human_readable=True):
        featmap = []
        for f in self.features:
            numcol = 1
            if len(f.output_shape) > 0:
                numcol = f.output_shape[0]
            for num in range(0, numcol):
                if human_readable:
                    featmap.append(f.name + "_" + str(num))
                else:
                    featmap.append(f.name)
        return featmap


    def add_feature(self, feature, oldmeta=None):
        if feature.name in self.featurestats:
            return
        self.features.append(feature)
        self.featurestats[feature.name] = dict()
        if len(feature.output_shape) <= 2:
            if oldmeta is not None and feature.name in oldmeta.featurestats and 'top10values' in \
                    oldmeta.featurestats[feature.name]:
                self.featurestats[feature.name]['top10values'] = oldmeta.featurestats[feature.name]['top10values']
            else:
                self.featurestats[feature.name]['top10values'] = [Counter() for _ in range(0, feature._get_width())]

    def add_label(self, label, oldmeta = None):
        if label.name in self.labelstats:
            return
        self.labels.append(label)
        self.labelstats[label.name] = dict()
        if len(label.output_shape) <= 2:
            if oldmeta is not None and label.name in oldmeta.labelstats and 'top10values' in \
                    oldmeta.labelstats[label.name]:
                self.labelstats[label.name]['top10values'] = oldmeta.labelstats[label.name]['top10values']
            else:
                self.labelstats[label.name]['top10values'] = [Counter() for _ in range(0, label._get_width())]

    def update_stats(self, feature_or_label_name, x):
        if feature_or_label_name in self.featurestats \
                and 'top10values' in self.featurestats[feature_or_label_name]:
            top10 = self.featurestats[feature_or_label_name]['top10values']
        elif feature_or_label_name in self.labelstats \
                and 'top10values' in self.labelstats[feature_or_label_name]:
            top10 = self.labelstats[feature_or_label_name]['top10values']
        else:
            return

        for i, cntr in enumerate(top10):
            if len(top10) == 1:
                cntr.update(x.flatten())
            else:
                cntr.update(x[:,i])
            if len(cntr) > 100:
                top10[i] = Counter(dict(cntr.most_common(100)))

    def load(self, fd):
        d = json.load(fd)
        for f in d['features']:
            feat = BaseFeature()
            feat.name = f['name']
            feat.dtype = f['dtype']
            feat.output_shape = tuple(f['output_shape'])
            self.features.append(feat)
        for f in d['labels']:
            feat = BaseFeature()
            feat.name = f['name']
            feat.dtype = f['dtype']
            feat.output_shape = tuple(f['output_shape'])
            self.labels.append(feat)
        self.featurestats = d['featurestats']
        for k,v in self.featurestats.items():
            if 'top10values' in v:
                for i, stat in enumerate(v['top10values']):
                    cntr_dict = dict()
                    for xx, yy in stat.items():
                        cntr_dict[float(xx)] = yy
                    v['top10values'][i] = Counter(cntr_dict)
        self.labelstats = d['labelstats']
        for k,v in self.labelstats.items():
            if 'top10values' in v:
                for i, stat in enumerate(v['top10values']):
                    cntr_dict = dict()
                    for xx, yy in stat.items():
                        cntr_dict[float(xx)] = yy
                    v['top10values'][i] = Counter(cntr_dict)
        self.model_input_shape = tuple(d['model_input_shape'])
        self.label_shape = tuple(d['label_shape'])
        self.sampletype = d['sampletype']

    def save(self, fd):
        d = {
            'featurestats': {},
            'labelstats': {},
            'model_input_shape': self.model_input_shape,
            'label_shape': self.label_shape,
            'sampletype': self.sampletype,
            'features': [],
            'labels': []
        }

        for k, v in self.featurestats.items():
            d['featurestats'][k] = v.copy()
            if 'top10values' in d['featurestats'][k]:
                d['featurestats'][k]['top10values'] = [dict(x.most_common(10)) for x in v['top10values']]

        for k,v in self.labelstats.items():
            d['labelstats'][k] = v.copy()
            if 'top10values' in d['labelstats'][k]:
                d['labelstats'][k]['top10values'] = [dict(x.most_common(10)) for x in v['top10values']]

        for feat in self.features:
            d['features'].append({'name': feat.name, 'dtype': str(feat.dtype), 'output_shape': feat.output_shape})
        for feat in self.labels:
            d['labels'].append({'name': feat.name, 'dtype': str(feat.dtype), 'output_shape': feat.output_shape})
        json.dump(d, fd)

    def _estimate_total_rows(self):
        """We don't want to save exact number of rows in the meta to avoid possible
        discrepancy between the number stored here and the actual number of rows stored in the parts."""

        rows = 0
        for x in self.featurestats.values():
            if 'top10values' in x:
                for y in x['top10values']:
                    r = sum(dict(y.most_common(1000)).values())
                    if r>rows:
                        rows = r
        for x in self.labelstats.values():
            if 'top10values' in x:
                for y in x['top10values']:
                    r = sum(dict(y.most_common(1000)).values())
                    if r>rows:
                        rows = r
        return rows


    def print_feature_summary(self):
        totalrows = self._estimate_total_rows()
        for k, v in self.featurestats.items():
            if 'top10values' not in v:
                logging.info('Not statistics about feature %s' % k)
                continue
            for i, vv in enumerate(v['top10values']):
                f_and_i = k
                if len(v['top10values']) > 1:
                    f_and_i += ', #' + str(i)
                if len(vv) == 0:
                    logging.warning('Feature %s has no values' % f_and_i)
                elif len(vv) == 1:
                    if vv.most_common(1)[0][0] == BaseFeature.MISSING_VALUE:
                        logging.warning('Feature %s has only missing values' % f_and_i)
                    else:
                        logging.warning('Feature %s has only one value %s' % (f_and_i, vv.most_common(1)[0][0]))
                else:
                    logging.info('Feature %s top 10 values are:' % (f_and_i))
                    for t in vv.most_common(10):
                        logging.info('%f: %d times' % t)
                    if vv.most_common(1)[0][1] > 0.99 * totalrows:
                        logging.warning('Feature %s has the value %s in more than 99%% of samples' % (f_and_i,
                                                                                                  vv.most_common(1)[0][
                                                                                                      0]))

    def _estimate_num_classes(self):
        if len(self.labels) != 1 or self.labels[0].output_shape != ():
            return None
        else:
            if 'top10values' not in self.labelstats[self.labels[0].name]:
                return None
            cntr = self.labelstats[self.labels[0].name]['top10values'][0]
            if len(cntr) == 0:
                raise Exception('Cannot estimate number of classes.')
            return int(np.max([x[0] for x in cntr.most_common(1000)])) + 1

    def print_label_summary(self):
        totalrows = self._estimate_total_rows()
        numclasses = self._estimate_num_classes()
        for k, v in self.labelstats.items():
            if 'top10values' not in v:
                logging.info('Not statistics about label %s' % k)
                continue
            for i, vv in enumerate(v['top10values']):
                if len(v['top10values']) == 1 and numclasses is not None:
                    map = dict(vv.most_common(1000))
                    for klass in range(0, numclasses):
                        if klass not in map:
                            logging.warning('Class %d is not present in label %s' %(klass, k))
                        elif map[klass] < 0.01*totalrows:
                            logging.warning('Class %d contributes to less than 1%% of the dataset' % klass)
                logging.info('Label %s top 10 values are:' % (k))
                for t in vv.most_common(10):
                    logging.info('%f: %d times' % t)

class DataSet(object):
    """
    Note that DataSet does not maintain the order of samples.

    """
    MAX_FILES_PER_DIR = 10000
    _test_global_remove_parts_setting = None

    @staticmethod
    def _getaveragedocsize(model, customclient=None):
        if customclient is None:
            client = mongoclient()
        else:
            client = customclient
        try:
            stats = client['IWLearn'].command('collstats', model.sampletype.__name__ + 's')
            return stats['avgObjSize'] * 2.0 if 'avgObjSize' in stats else 100000
        finally:
            if customclient is None:
                client.close()

    @staticmethod
    def remove(experiment_name):
        try:
            shutil.rmtree('input/' + experiment_name)
        except:
            pass

    @staticmethod
    def generate(experiment_name, model, maxRAM=None, part_size=None, customclient=None, **kwargs):
        """
        Generate new or extend existing dataset by loading it from MongoDB, and cache it on disk. The disk cache will
        be split in parts samplewise, each part containing only part_size of samples. Inside of each part, one file
        per feature will be saved, in the .npy (numpy.save format).

        The separation into parts allows both for generating datasets larger than RAM as well as reading from such
        datasets during the training.

        The saving of features into separate files allows for easy extension of removal of features for existing
        datasets.

        :param experiment_name: The name of the subdirectory containing cached dataset.
        :param model: The model this dataset has to be created for. The model defines both the features that are
        to be extracted from the samples, as well as the shape of the input matrix. Also, the model defines the
        type of samples to load in case the query parameter is passed.
        :param maxRAM: maximum RAM to use for generating the dataset. If you don't pass batch_size in kwargs, the
        average sample size will be calculated and maxRAM will be used to determine maximal batch_size fitting into
        the maxRAM limit. If maxRAM is not passed, we take 50% of all RAM on this PC.
        :param part_size: pass number of samples to be contained within each part of dataset separately written to
        disk. Usually you don't need to care about it, as the part size will be chosen automatically. In case your
        samples are extremely small or extremely big though, you might need to tweak this parameter. We find the parts
        of below 4 MiB in byte size to work best.
        :param customclient: optionally, a mongo client to use (for example if you want to pass a mock client for tests)
        :param **kwargs: pass arguments to the mongo client find method to load the samples, eg. filter, batch_size or
        projection
        :return: a new DataSet instance
        """

        if customclient is None:
            client = mongoclient()
        else:
            client = customclient

        try:
            coll = client['IWLearn'][model.sampletype.__name__ + 's']

            if 'batch_size' not in kwargs:
                if maxRAM is None:
                    maxRAM = int(0.5 * os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))

                average_doc_size = DataSet._getaveragedocsize(model, client)
                batch_size = int(maxRAM / average_doc_size)

            if 'filter' in kwargs:
                if 'batch_size' not in kwargs:
                    kwargs['batch_size'] = batch_size
                cursor = coll.find(**kwargs)
            elif 'pipeline' in kwargs:
                if 'cursor' not in kwargs:
                    kwargs['cursor'] = {'batchSize': batch_size}

                cursor = coll.aggregate(**kwargs)
            else:
                raise Exception('provide filter or pipeline')

            logging.info('Determined batch_size is %d' % batch_size)

            if DataSet._generateImpl(
                    experiment_name,
                    model,
                    lambda: model.sampletype.fromjson(next(cursor)),
                    part_size) == 0:
                raise Exception('no samples')
        except:
            logging.error('Cannot generate dataset')
        finally:
            if customclient is None and client is not None:  #
                client.close()

    @staticmethod
    def _get_optimal_nesting(numparts):
        if numparts == 0:
            return []
        k = 1
        while True:
            x = pow(numparts, 1.0 / k)
            if x < DataSet.MAX_FILES_PER_DIR:
                break
            k += 1
        L = int(ceil(log(numparts / x, 16)))
        return [L] * (k - 1)

    @staticmethod
    def bootstrap(experiment_name, model, samples, part_size=None):
        """
        Generate new or extend existing dataset by bootstrapping (i.e. creating fake samples from some pre-existing
        data), and cache it on disk. The disk cache will be split in parts samplewise, each part containing only
        part_size of samples. Inside of each part, one file per feature will be saved, in the .npy (numpy.save format).

        The separation into parts allows both for generating datasets larger than RAM as well as reading from such
        datasets during the training.

        The saving of features into separate files allows for easy extension of removal of features for existing
        datasets.

        :param experiment_name: The name of the subdirectory containing cached dataset.
        :param model: The model this dataset has to be created for. The model defines both the features that are
        to be extracted from the samples, as well as the shape of the input matrix. Also, the model defines the
        type of samples to load in case the query parameter is passed.
        :param samples: the list or iterable delivering samples
        :param part_size: pass number of samples to be contained within each part of dataset separately written to
        disk. Usually you don't need to care about it, as the part size will be chosen automatically. In case your
        samples are extremely small or extremely big though, you might need to tweak this parameter. We find the parts
        of below 4 MiB in byte size to work best.
        """

        iterator = samples.__iter__()

        nesting = []
        if part_size is not None:
            nesting = DataSet._get_optimal_nesting(len(samples) / part_size)

        if DataSet._generateImpl(
                experiment_name,
                model,
                lambda: next(iterator),
                part_size,
                nesting) == 0:
            raise Exception('Cannog generate set: no samples')

    @staticmethod
    def fname(featurename):
        return re.sub(r'[\W_]+', '', featurename) + '.npy'

    @staticmethod
    def _generateImpl(experiment_name, model, samplefetcher, part_size, nesting=[]):
        # only get part_size samples at a time, convert and save them to disk, then repeat
        # this is to process data sets where all samples do not fit into the RAM

        datasetmeta = DataSetMeta()
        oldmeta = None

        if os.path.isdir('input/' + experiment_name):
            if os.path.isfile('input/%s/dataset_V8.json' % experiment_name):
                with open('input/%s/dataset_V8.json' % experiment_name, 'r') as f:
                    oldmeta = DataSetMeta()
                    oldmeta.load(f)
            else:
                raise Exception('Directory input/%s is present, but there is no file dataset_V8.json inside. '
                                'It looks like your previous generation or bootstrapping attempt has failed.'
                                'Please cleanup the directory manually.'
                                'Dataset generation aborted.' % experiment_name)
        else:
            os.makedirs('input/' + experiment_name)

        for feature in model.features:
            datasetmeta.add_feature(feature, oldmeta)

        for label in model.labels:
            datasetmeta.add_label(label, oldmeta)

        datasetmeta.sampletype = model.sampletype.__name__ if model.sampletype is not None else None
        datasetmeta.model_input_shape = model.input_shape
        datasetmeta.label_shape = model.output_shape

        if oldmeta is not None:
            removed_features = set(oldmeta.featurestats.keys()).difference(datasetmeta.featurestats.keys())
            if len(removed_features) > 0:
                logging.warning('Following features removed from dataset_V8.json: %s' % (str(removed_features)))

            removed_labels = set(oldmeta.labelstats.keys()).difference(datasetmeta.labelstats.keys())
            if len(removed_labels) > 0:
                logging.warning('Following labels removed from dataset_V8.json: %s' % (str(removed_labels)))

            if oldmeta.sampletype != datasetmeta.sampletype:
                logging.warning('Sample type has been changed from %s to %s' % (oldmeta.sampletype, datasetmeta.sampletype))

            if oldmeta.model_input_shape != datasetmeta.model_input_shape:
                logging.warning('Model input shape has been changed from %s to %s' % (oldmeta.model_input_shape,
                                                                      datasetmeta.model_input_shape))

            if oldmeta.label_shape != datasetmeta.label_shape:
                logging.warning('Label shape has been changed from %s to %s' % (oldmeta.label_shape,
                                                                           datasetmeta.label_shape))

        if part_size is None:
            def part_size_estimator(sample):
                samplebytesize = 0
                for f in model.features:
                    x = create_tensor([sample], f, preprocess=False)
                    samplebytesize += sys.getsizeof(x)
                for f in model.labels:
                    x = create_tensor([sample], f, preprocess=False)
                    samplebytesize += sys.getsizeof(x)
                part_size = int(4000000.0 / samplebytesize)
                if part_size < 1:
                    part_size = 1
                logging.info('Dataset part_size calculated to be %d' % part_size)
                return part_size

            part_size = part_size_estimator

        def featuretensorgetter(samples, featurename):
            feature = None
            for f in model.features:
                if f.name == featurename:
                    feature = f
                    break
            for f in model.labels:
                if f.name == featurename:
                    feature = f
                    break
            return create_tensor(samples, feature)

        persisters = dict()
        for feature in model.features:
            if hasattr(feature, 'persister'):
                persisters[feature.name] = feature.persister
        for feature in model.labels:
            if hasattr(feature, 'persister'):
                persisters[feature.name] = feature.persister

        return DataSet._write(
            datasetmeta,
            experiment_name,
            [f.name for f in model.features],
            [f.name for f in model.labels],
            nesting,
            part_size,
            samplefetcher,
            featuretensorgetter,
            persisters)

    @staticmethod
    def _write(
            datasetmeta,
            experiment_name,
            feature_names,
            label_names,
            nesting,
            part_size,
            samplefetcher,
            tensorgetter,
            persisters={}):
        totalrows = 0
        eof = False
        while not eof:
            samples = []
            hashvariable = sha256()
            ids = []

            if callable(part_size):
                # Estimate sample size and calculate optimal part size
                try:
                    sample = samplefetcher()
                except StopIteration:
                    raise Exception('Trying to generate an empty dataset')
                sampleid = str(sample.entityid)
                ids.append(sampleid)
                hashvariable.update(sampleid.encode('utf8'))
                samples.append(sample)
                part_size = part_size(sample)

            for _ in range(0, part_size):
                try:
                    sample = samplefetcher()
                    sampleid = str(sample.entityid)
                    ids.append(sampleid)
                    hashvariable.update(sampleid.encode('utf8'))
                    samples.append(sample)
                except StopIteration:
                    eof = True
                    break
                except:
                    logging.exception('cannot fetch sample')

            if len(samples) == 0:
                break

            if Settings.EnsureUniqueEntitiesInDataset and len(ids) > len(set(ids)):
                raise Exception('String representations of sample ids are not unique')

            totalrows += len(samples)

            digest = hashvariable.hexdigest()
            partdir = 'input/%s' % experiment_name
            h_idx = 0
            for nest in nesting:
                partdir += '/' + digest[h_idx: h_idx + nest]
                h_idx += nest
            partdir += '/part_%s' % digest

            if os.path.isdir(partdir):
                with open(partdir + '/part.json', 'r') as f:
                    partmeta = json.load(f)
                partexists = True
            else:
                partexists = False
                partmeta = {
                    'bytesize': 0,
                    'numsamples': len(samples),
                    'unordered_features': []
                }
                os.makedirs(partdir)

            if partexists:
                # because conversion from sample to X takes time, we don't perform it, if there is already a cached
                # part on the disk. This is especially handy in the case when dataset processing had terminated due to
                # a bug in some feature, so you have to restart it.
                features_to_get = []
                for feature_name in feature_names:
                    featurefile = '%s/%s' % (partdir, DataSet.fname(feature_name))
                    if not os.path.isfile(featurefile):
                        features_to_get.append(feature_name)
            else:
                features_to_get = feature_names

            if len(features_to_get) > 0:
                for feature_name in features_to_get:
                    featurefile = '%s/%s' % (partdir, DataSet.fname(feature_name))

                    x = tensorgetter(samples, feature_name)
                    x[np.isnan(x)] = BaseFeature.MISSING_VALUE

                    datasetmeta.update_stats(feature_name, x)

                    if feature_name in persisters:
                        persisters[feature_name].save(featurefile, x)
                    else:
                        with open(featurefile, 'wb') as f:
                            np.save(f, x)

                    if feature_name not in partmeta['unordered_features']:
                        partmeta['unordered_features'].append(feature_name)

                    partmeta['bytesize'] += sys.getsizeof(x)

            for label_name in label_names:
                labelfile = '%s/Label-%s' % (partdir, DataSet.fname(label_name))

                x = tensorgetter(samples, label_name)
                x[np.isnan(x)] = BaseFeature.MISSING_VALUE

                datasetmeta.update_stats(label_name, x)

                if label_name in persisters:
                    persisters[label_name].save(labelfile, x)
                else:
                    with open(labelfile, 'wb') as f:
                        np.save(f, x)

                partmeta['bytesize'] += sys.getsizeof(x)

            if not os.path.isfile(partdir + '/ids.txt'):
                with open(partdir + '/ids.txt', 'wb') as f:
                    f.writelines([(x + "\n").encode('utf8') for x in ids])

            with open(partdir + '/part.json', 'w') as f:
                json.dump(partmeta, f)
            logging.info('%s stored or updated. In total %d rows generated' % (partdir, totalrows))
            with open('input/%s/dataset_V8.json' % experiment_name, 'w') as f:
                datasetmeta.save(f)

        sollfeatures = set(datasetmeta.featurestats.keys())
        for entry in scandir('input/%s' % experiment_name):
            if entry.is_dir() and entry.name.startswith('part_'):
                metafile = 'input/%s/%s/part.json' % (experiment_name, entry.name)
                if os.path.isfile(metafile):
                    with open(metafile, 'r') as f:
                        meta = json.load(f)
                        ist = set(meta['unordered_features'])
                        missing = sollfeatures.difference(ist)
                        if len(missing) > 0:
                            logging.warning('%s does not contain following features: %s ' % (entry, str(missing)))
                            if DataSet._test_global_remove_parts_setting is not None:
                                x = DataSet._test_global_remove_parts_setting
                            else:
                                x = input(
                                    'Press y to remove the part, any other key to leave it (in this case missing feature '
                                    'will always have missing values)')
                            if x == 'y':
                                shutil.rmtree('input/%s/%s' % (experiment_name, entry))

        datasetmeta.print_feature_summary()
        datasetmeta.print_label_summary()

        return totalrows

    @staticmethod
    def optimize(experiment_name):
        ds = DataSet(experiment_name)
        if len(ds.metaparts) > DataSet.MAX_FILES_PER_DIR:
            nesting = DataSet._get_optimal_nesting(len(ds.metaparts))
            logging.info('Optimizing dataset by introducing nesting ' + str(nesting))
            for olddir in tqdm(ds.partmap_names[1:]):
                digest = olddir.split('/')[-1][len('part_'):]
                partdir = 'input/%s' % experiment_name
                h_idx = 0
                for nest in nesting:
                    partdir += '/' + digest[h_idx: h_idx + nest]
                    h_idx += nest
                try:
                    os.makedirs(partdir)
                except:
                    pass
                partdir += '/part_%s' % digest
                if olddir != partdir:
                    shutil.move(olddir, partdir)

            logging.info('Removing empty directories')
            for entry in scandir('input/%s' % experiment_name):
                if entry.is_dir():
                    if len(scandir(entry.path)) == 0:
                        shutil.rmtree(entry.path)

    @staticmethod
    def exists(experiment_name):
        return os.path.isfile('input/%s/dataset_V8.json' % experiment_name)

    def __init__(self, experiment_name, maxRAM=None):
        self.experiment_name = experiment_name

        self.meta = None

        if os.path.isfile('input/%s/dataset_V8.json' % experiment_name):
            with open('input/%s/dataset_V8.json' % experiment_name, 'r') as f:
                self.meta = DataSetMeta()
                self.meta.load(f)
        else:
            raise Exception('input/%s/dataset_V8.json not found, please enter right path or '
                            'generate the dataset first' % experiment_name)

        logging.info('Loading dataset %s' % experiment_name)

        self.average_feature_size = 0
        self.numsamples = 0
        self.partmap_indexes = [0]
        self.partmap_names = [None]
        self.model_input_shape = tuple(self.meta.model_input_shape)
        self.label_shape = tuple(self.meta.label_shape)
        self.strsampletype = self.meta.sampletype

        self.metaparts = {}
        self.cache = None
        bytesize = self._load_metaparts('input/%s' % experiment_name)
        if len(self.metaparts) > 0:
            self.average_feature_size = bytesize * 1.0 / len(self.metaparts) / (len(self.meta.features)
                                                                                + len(self.meta.labels))
            logging.info('Average feature size %f' % self.average_feature_size)

            if maxRAM is None:
                maxRAM = int(0.5 * os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))

            if self.average_feature_size > 0:
                maxfeatures = maxRAM / self.average_feature_size
            else:
                maxfeatures = len(self.metaparts) * (len(self.meta.features)
                                                                                + len(self.meta.labels))

            maxfeatures = int(maxfeatures)
            if maxfeatures == 0:
                maxfeatures = 1

            logging.info('Creating dataset cache storing at most %d features/labels' % maxfeatures)
            self.cache = LRU(int(maxfeatures))

    def _load_metaparts(self, dir):
        bytesize = 0
        for entry in scandir(dir):
            if entry.is_dir():
                if entry.name.startswith('part_'):
                    try:
                        with open('%s/part.json' % entry.path, 'r') as f:
                            partmeta = json.load(f)
                            self.metaparts[entry.path] = partmeta
                            self.numsamples += partmeta['numsamples']
                            self.partmap_indexes.append(self.numsamples)
                            bytesize += partmeta['bytesize']
                            self.partmap_names.append(entry.path)
                    except:
                        logging.error('Cannot load part %s' % entry.path)
                else:
                    self._load_metaparts(entry.path)
        return bytesize

    def _load_x(self, partdir, name, islabel):
        if islabel:
            f_file = partdir + 'Label-' + DataSet.fname(name)
        else:
            f_file = partdir + DataSet.fname(name)
        if f_file in self.cache:
            return self.cache[f_file]

        x = None
        if os.path.isfile(f_file + '.persister'):
            with open(f_file + '.persister', 'rb') as pf:
                persister = pickle.load(pf)
            x = persister.load(f_file)
        elif os.path.isfile(f_file):
            x = np.load(f_file)
        else:
            logging.warning('%s does not contain feature %s, replacing with missing value' % (partdir, f_file))

        self.cache[f_file] = x
        return x


    def _load_features(self, partname):
        partdir = partname + '/'

        feat = {}
        for ff in self.meta.features:
            x = self._load_x(partdir, ff.name, False)
            if x is None:
                if Settings.OnFeatureMissing == 'raise':
                    raise Exception(
                        'Feature file %s is missing in the part %s. If you want its values to be replaced with '
                        'MISSING_VALUE, set Settings.OnFeatureMissing to impute' % (
                        ff.name, partdir))
                else:
                    logging.info('Feature file %s is missing in the part %s.' % (ff.name, partdir))
                    x = np.full((self.metaparts[partname]['numsamples'],) + tuple(ff.output_shape),
                                BaseFeature.MISSING_VALUE, dtype=ff.dtype)
            feat[ff.name] = x

        return feat

    def _load_labels(self, partname):
        partdir = partname + '/'

        labe = {}
        for ff in self.meta.labels:
            labe[ff.name] = self._load_x(partdir, ff.name, True)

        return labe

    def _get_part_and_offset(self, sample_index):
        i = bisect_right(self.partmap_indexes, sample_index)
        if i > 0:
            offset = sample_index - self.partmap_indexes[i - 1]
            part, offset = self.partmap_names[i], offset
            return part, offset
        raise ValueError()

    def get_sample_id(self, sample_index):
        partname, offset = self._get_part_and_offset(sample_index)
        with open(partname + '/ids.txt', 'r') as f:
            ids = f.readlines()
            return ids[offset][:-1]

    def __getitem__(self, index):
        """
        Return a tuple for one training sample containing:
         1) X with shape (1,) + self.model_input_shape
         2) y_true with shape (1,) + self.label_shape

        Use the method get_all_samples instead, which is more efficient if all your
        samples fit into RAM.
        """
        if len(self.meta.labels) > 1:
            raise Exception('Use get_one_sample instead of getitem, because you have to pass the label')

        featuremetas, labelmeta, x_shape, y_shape = self._getmetas()
        if x_shape != self.model_input_shape:
            raise Exception('Features of the dataset have shape %s, but model expect input shape %s' % (
            x_shape, self.model_input_shape))

        if y_shape is not None and y_shape != self.label_shape:
            raise Exception('Labels of the dataset have shape %s, but model delivers output of shape %s' % (
            y_shape, self.label_shape))

        return self.get_one_sample(index, featuremetas, labelmeta, x_shape)

    def get_one_sample(self, index, featuremetas, labelmeta, x_shape):
        if index >= self.numsamples:
            raise IndexError()

        partname, offset = self._get_part_and_offset(index)
        feat = self._load_features(partname)
        labe = self._load_labels(partname)

        x = combine_tensors([feat[ff.name][offset:offset + 1] for ff in featuremetas], x_shape, 1)[0]
        if labelmeta:
            y = labe[labelmeta.name][offset]

        if labelmeta:
            return x, y
        else:
            return x, None

    def __len__(self):
        return self.numsamples

    def _getmetas(self, features=None, label=None):
        if features is None:
            featuremetas = self.meta.features
        else:
            featuremetas = [ff for ff in self.meta.features if ff.name in features]
            if len(featuremetas) < len(features):
                raise Exception('Dataset %s does not contain the following features: %s' % (
                self.experiment_name, ','.join(set(features).difference(set([ff.name for ff in featuremetas])))))

        if label is None:
            if len(self.meta.labels) > 1:
                raise Exception('Specify which label should be retrieved from the dataset')
            if len(self.meta.labels) > 0:
                labelmeta = self.meta.labels[0]
            else:
                labelmeta = None
        else:
            labelmeta = [ff for ff in self.meta.labels if ff.name == label]
            if len(labelmeta) == 0:
                raise Exception('Dataset does not contain label %s' % label)
            labelmeta = labelmeta[0]

        x_shape = BaseModel.calculate_shape(shapes=[ff.output_shape for ff in featuremetas])
        y_shape = tuple(labelmeta.output_shape) if labelmeta is not None else None

        return featuremetas, labelmeta, x_shape, y_shape

    def get_all_samples(self, features=None, label=None):
        """
        Return all samples of the dataset - you should always use this method if your dataset fits into memory.
        If you don't pass features, all features of the dataset will be returned. If the dataset has more than
        one label, you must pass the label to be used
        :return: A tuple of (X, y_true)
        """

        featuremetas, labelmeta, x_shape, y_shape = self._getmetas(features, label)
        if features is None and x_shape != self.model_input_shape:
            raise Exception('Features of the dataset have shape %s, but model expect input shape %s' % (
            x_shape, self.model_input_shape))

        if len(self.meta.labels) == 1 and y_shape != self.label_shape:
            raise Exception('Label of the dataset has shape %s, but model delivers output of shape %s' % (
            y_shape, self.label_shape))

        all_x = np.zeros((self.numsamples,) + x_shape)
        if labelmeta:
            all_y = np.zeros((self.numsamples,) + y_shape)

        row = 0
        for partname in self.partmap_names[1:]:
            feat = self._load_features(partname)
            labe = self._load_labels(partname)
            numsamples = self.metaparts[partname]['numsamples']
            all_x[row:row + numsamples] = combine_tensors([feat[ff.name] for ff in featuremetas], x_shape,
                                                          numsamples)
            if labelmeta:
                all_y[row:row + numsamples] = labe[labelmeta.name]
            row += numsamples

        if labelmeta:
            return all_x, all_y
        else:
            return all_x, None

    def get_all_labels(self, labels=None):
        """
        Return all labels of the dataset
        :return: y_true
        """

        featuremetas, labelmeta, x_shape, y_shape = self._getmetas([], labels)
        all_y = np.zeros((self.numsamples,) + y_shape)

        row = 0
        for partname in self.partmap_names[1:]:
            labe = self._load_labels(partname)
            numsamples = self.metaparts[partname]['numsamples']
            all_y[row:row + numsamples] = labe[labelmeta.name]
            row += numsamples

        return all_y


    def get_samples(self, index_array, features=None, label=None):
        featuremetas, labelmeta, x_shape, y_shape = self._getmetas(features, label)
        batch_x = []
        batch_y = []

        for i, j in enumerate(index_array):
            x, y = self.get_one_sample(j, featuremetas, labelmeta, x_shape)
            batch_x.append(x)
            if labelmeta:
                batch_y.append(y)

        if labelmeta:
            return np.stack(batch_x), np.stack(batch_y)
        else:
            return np.stack(batch_x), None

    def getfeaturemap(self, human_readable=True):
        return self.meta.getfeaturemap(human_readable)

    def clone(self, experiment_name, index_array=None, processor=None, persisters={}):
        if index_array is None:
            index_array = np.arange(0, len(self))
        if processor is None:
            processor = lambda feature_name, X: X


        if os.path.isdir('input/' + experiment_name):
            raise Exception('Directory input/%s is already present, cannot clone into that'
                                 % experiment_name)
        else:
            os.makedirs('input/' + experiment_name)

        idx_iter = index_array.__iter__()

        part_size = list(self.metaparts.values())[0]['numsamples']
        nesting = DataSet._get_optimal_nesting(len(index_array) / part_size)
        feature_names = [x.name for x in self.meta.features]
        label_names = [x.name for x in self.meta.labels]

        class TensorCarrier(BaseSample):
            def __init__(self, entityid, tensors):
                BaseSample.__init__(self, entityid)
                self.tensors = tensors

        def clonesamplefetcher():
            idx = next(idx_iter)
            tensors = {}
            for f in feature_names:
                featuremetas, _, _, _ = self._getmetas(features=[f])
                X, _ = self.get_one_sample(idx, featuremetas, None, tuple(featuremetas[0].output_shape))
                tensors[f] = processor(f, X)
            for f in label_names:
                _, labelmeta, _, _ = self._getmetas(label=f)
                _, y = self.get_one_sample(idx, [], labelmeta, ())
                tensors[f] = processor(f, y)
            sampleid = self.get_sample_id(idx)
            return TensorCarrier(sampleid, tensors)

        def clonetensorgetter(samples, featurename):
            return np.stack([x.tensors[featurename] for x in samples])

        newmeta = self.meta.copy(reset_stats=True)

        DataSet._write(
            newmeta,
            experiment_name,
            feature_names,
            label_names,
            nesting,
            part_size,
            clonesamplefetcher,
            clonetensorgetter,
            persisters
        )

    def unsafe_rewrite_dataset(self, experiment_name, X, y, sampleids):
        if X.shape[1:] != tuple(self.meta.model_input_shape):
            raise Exception('X.shape %s does not correspond to dataset model_input_shape %s' % (
            X.shape[1:], self.meta.model_input_shape))
        if y.shape[1:] != tuple(self.meta.label_shape):
            raise Exception(
                'y.shape %s does not correspond to dataset label shape %s' % (y.shape[1:],
                                                                              self.meta.label_shape))
        if len(X) != len(y):
            raise Exception('Lengths of X and y do not agree, %d != %d' % (len(X), len(y)))

        zip_iterator = zip(X, y, sampleids).__iter__()

        part_size = self.metaparts.values()[0]['numsamples']
        nesting = DataSet._get_optimal_nesting(len(X) / part_size)
        feature_names = [x.name for x in self.meta.features]
        label_names = [x.name for x in self.meta.labels]

        class TensorCarrier(BaseSample):
            def __init__(self, entityid, tensors):
                BaseSample.__init__(self, entityid)
                self.tensors = tensors

        def rewritesamplefetcher():
            one_X, one_y, one_sampleid = next(zip_iterator)
            tensors = {}
            for col, f in enumerate(feature_names):
                featuremetas, _, _, _ = self._getmetas(features=[f])
                width = featuremetas[0]['output_shape']
                if len(width) == 0:
                    width = 1
                else:
                    width = width[0]
                tensors[f] = one_X[col:col + width]
            for col, f in enumerate(label_names):
                _, labelmeta, _, _ = self._getmetas(label=f)
                width = labelmeta['output_shape']
                if len(width) == 0:
                    tensors[f] = one_y
                else:
                    width = width[0]
                    tensors[f] = one_y[col:col + width]
            sampleid = self.get_sample_id(one_sampleid)
            return TensorCarrier(sampleid, tensors)

        def rewritetensorgetter(samples, featurename):
            return np.stack([x.tensors[featurename] for x in samples])

        newmeta = self.meta.copy(reset_stats=True)

        DataSet._write(
            newmeta,
            experiment_name,
            feature_names,
            label_names,
            nesting,
            part_size,
            rewritesamplefetcher,
            rewritetensorgetter
        )

    def plot_data_classification(self, bins=10, all_in_one=True, index_array=None):
        num_charts = self.model_input_shape[0] + 1
        r = floor(sqrt(num_charts)) + 1
        c = ceil(sqrt(num_charts))

        map = self.getfeaturemap()
        map.append('Label')

        if index_array is None:
            X, y_true = self.get_all_samples()
        else:
            X, y_true = self.get_samples(index_array)

        for column_index, featurename in enumerate(map):
            if all_in_one:
                plt.subplot(r, c, column_index + 1)
            if column_index == len(map) - 1:
                allvalues = y_true
            else:
                allvalues = X[:, column_index]
            allmin = np.min(allvalues)
            allmax = np.max(allvalues)
            bar_width = 0
            numclasses = self.meta._estimate_num_classes()
            for class_index in range(0, numclasses):
                h, w = np.histogram(
                    allvalues[y_true == class_index] if numclasses > 1 else allvalues,
                    density=True,
                    bins=bins,
                    range=(allmin, 1 + allmax))
                bar_width = max(abs(w)) / (4.0 * bins)
                if class_index == 0:
                    plt.bar(w[0:-1], h, bar_width, color='g', alpha=0.7)
                elif class_index == 1:
                    plt.bar(w[0:-1] + bar_width, h, bar_width, color='r', alpha=0.7)
                else:
                    plt.bar(w[0:-1] + bar_width * column_index, h, bar_width, alpha=0.7)

            plt.xticks(w[0:-1] + bar_width * numclasses / 2, [float(round(x, 2)) for x in w[0:-1]])
            plt.title(featurename)
            if not all_in_one:
                plt.show()

        if all_in_one:
            plt.subplots_adjust(hspace=0.4)
            plt.show()

    def plot_data_regression(self, all_in_one=True, index_array=None):
        num_charts = self.model_input_shape[0] + 1
        r = floor(sqrt(num_charts)) + 1
        c = ceil(sqrt(num_charts))

        map = self.getfeaturemap()

        if index_array is None:
            X, y_true = self.get_all_samples()
        else:
            X, y_true = self.get_samples(index_array)

        for column_index, featurename in enumerate(map):
            if all_in_one:
                plt.subplot(r, c, column_index + 1)
            allvalues = X[:, column_index]
            plt.scatter(allvalues, y_true)
            plt.title(featurename)
            if not all_in_one:
                plt.show()

        if all_in_one:
            plt.subplots_adjust(hspace=0.4)
            plt.show()


class PillowPersister(object):
    def __init__(self, width, height, mode='RGB'):
        self.width = width
        self.height = height
        self.mode = mode
        self.numchannels = len(mode)

    def save(self, featurefile, x):
        for i, s in enumerate(x):
            img = Image.frombytes(self.mode, (self.width, self.height), s)
            with open('%s_%i.jpg' % (featurefile, i), 'wb') as f:
                img.save(f, format='JPEG')
        self.numimages = len(x)
        with open('%s.persister' % (featurefile), 'wb') as f:
            pickle.dump(self, f)

    def load(self, featurefile):
        x = np.full((self.numimages, self.width, self.height, self.numchannels), BaseFeature.MISSING_VALUE)
        for i in range(0, self.numimages):
            file = '%s_%i.jpg' % (featurefile, i)
            if os.path.isfile(file):
                img = Image.open(file)
                x[i] = np.asarray(img, dtype=np.uint8)
        return x


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # print (DataSet._get_optimal_nesting(10))
    # print (DataSet._get_optimal_nesting(1000))
    # print (DataSet._get_optimal_nesting(50000))
    print(DataSet._get_optimal_nesting(400000))
    # print (DataSet._get_optimal_nesting(1000000))
