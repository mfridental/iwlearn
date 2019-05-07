# -*- coding: utf-8 -*-
import cPickle
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


class DataSet(object):
    """
    Note that DataSet does not maintain the order of samples
    """
    MAX_FILES_PER_DIR = 10000

    @staticmethod
    def _getaveragedocsize(model, customclient=None):
        if customclient is None:
            client = mongoclient()
        else:
            client = customclient
        try:
            stats = client['IWLearn'].command('collstats', model.sampletype.__name__ + 's')
            return stats['avgObjSize']*2.0 if 'avgObjSize' in stats else 100000
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
    def generate(experiment_name, model, maxRAM=None, numclasses=None, part_size=None, customclient=None, **kwargs):
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
        :param numclasses: in case the data set is for a classifier, pass number of classes. This information cannot
        be retrieved from the model in case if model.output_shape = (1,).
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
                lambda: model.sampletype.fromjson(cursor.next()),
                part_size,
                numclasses) == 0:
                raise Exception('Cannot generate set: no samples')
        except Exception as e:
            logging.error(e.message)
        finally:
            if customclient is None and client is not None:#
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
        L = int(ceil(log(numparts/x, 16)))
        return [L]*(k-1)


    @staticmethod
    def bootstrap(experiment_name, model, samples, numclasses = None, part_size = None):
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
        :param numclasses: in case the data set is for a classifier, pass number of classes. This information cannot
        be retrieved from the model in case if model.output_shape = (1,).
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
            lambda: iterator.next(),
            part_size,
            numclasses,
            nesting) == 0:
            raise Exception('Cannog generate set: no samples')


    @staticmethod
    def fname(featurename):
        return re.sub(r'[\W_]+', '', featurename) + '.npy'

    @staticmethod
    def _generateImpl(experiment_name, model, samplefetcher, part_size, numclasses = None, nesting=[]):
        # only get part_size samples at a time, convert and save them to disk, then repeat
        # this is to process data sets where all samples do not fit into the RAM

        datasetmeta = {
            'features': [],
            'labels': [],
            'sampletype': model.sampletype.__name__ if model.sampletype is not None else None}

        if os.path.isdir('input/' + experiment_name):
            if os.path.isfile('input/%s/dataset_V5.json' % experiment_name):
                with open('input/%s/dataset_V5.json' % experiment_name, 'r') as f:
                    datasetmeta = json.load(f)
            else:
                raise Exception('Directory input/%s is present, but there is no file dataset_V5.json inside. '
                                'It looks like your previous generation or bootstrapping attempt has failed.'
                                'Please cleanup the directory manually.'
                                'Dataset generation aborted.' % experiment_name)
        else:
            os.makedirs('input/' + experiment_name)

        oldfeatures = dict([(x['name'].encode('utf8'), x) for x in datasetmeta['features']])
        datasetmeta['features'] = []

        for feature in model.features:
            if len(feature.output_shape) <= 2:
                numtop = feature._get_width()
            else:
                numtop = 1
            f = {
                    'name': feature.name,
                    'output_shape': feature.output_shape,
                    'top10values': [Counter() for _ in xrange(0, numtop)],
                    'dtype': str(feature.dtype)
             }
            if feature.name in oldfeatures:
                f['top10values'] = [Counter(x) for x in oldfeatures[feature.name]['top10values']]
            datasetmeta['features'].append(f)

        oldlabels = dict([(x['name'].encode('utf8'), x) for x in datasetmeta['labels']])
        datasetmeta['labels'] = []
        for label in model.labels:
            f = {
                    'name': label.name,
                    'output_shape': label.output_shape
             }
            datasetmeta['labels'].append(f)

        removed_features = set(oldfeatures.keys()).difference(set([x.name for x in model.features]))
        if len(removed_features) > 0:
            logging.warning('Following features removed from dataset_V5.json: %s' % (str(removed_features)))

        removed_labels = set(oldlabels.keys()).difference(set([x.name for x in model.labels]))
        if len(removed_labels) > 0:
            logging.warning('Following labels removed from dataset_V5.json: %s' % (str(removed_labels)))

        datasetmeta['model_input_shape'] = model.input_shape
        datasetmeta['label_shape'] = model.output_shape

        if model._get_width() == 1 and numclasses is None:
            raise Exception('Pass parameter numclasses for models with output_shape = (1,0)')

        if numclasses is None:
            numclasses = model._get_width()

        datasetmeta['numclasses'] = numclasses

        if numclasses > 0 and 'class_counts' not in datasetmeta:
            datasetmeta['class_counts'] = {}
        for cls in xrange(0, numclasses):
            klass = 'Class_' + str(cls)
            if klass not in datasetmeta['class_counts']:
                datasetmeta['class_counts'][klass] = 0

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
            numclasses,
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
            numclasses,
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
                hashvariable.update(sampleid)
                samples.append(sample)
                part_size = part_size(sample)

            for _ in xrange(0, part_size):
                try:
                    sample = samplefetcher()
                    sampleid = str(sample.entityid)
                    ids.append(sampleid)
                    hashvariable.update(sampleid)
                    samples.append(sample)
                except StopIteration:
                    eof = True
                    break

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
                for feature in feature_names:
                    featurefile = '%s/%s' % (partdir, DataSet.fname(feature))
                    if not os.path.isfile(featurefile):
                        features_to_get.append(feature)
            else:
                features_to_get = feature_names

            if len(features_to_get) > 0:
                for feature in features_to_get:
                    featurefile = '%s/%s' % (partdir, DataSet.fname(feature))

                    x = tensorgetter(samples, feature)
                    x[np.isnan(x)] = BaseFeature.MISSING_VALUE

                    try:
                        for ff in datasetmeta['features']:
                            if ff['name'] == feature:
                                if len(ff['output_shape']) == 0:
                                    cntr = ff['top10values'][0]
                                    cntr.update(x)
                                    if len(cntr) > 10:
                                        ff['top10values'][0] = Counter(dict(cntr.most_common(10)))
                                elif len(ff['output_shape']) == 1:
                                    for i in xrange(0, ff['output_shape'][0]):
                                        cntr = ff['top10values'][i]
                                        cntr.update(x[:, i])
                                        if len(cntr) > 10:
                                            ff['top10values'][i] = Counter(dict(cntr.most_common(10)))
                                else:
                                    cntr = ff['top10values'][0]
                                    cntr.update([np.mean(x)])
                                    if len(cntr) > 10:
                                        ff['top10values'][0] = Counter(dict(cntr.most_common(10)))

                                break
                    except:
                        logging.info('Cannot calculate top10values ' + traceback.format_exc())

                    if feature in persisters:
                        persisters[feature].save(featurefile, x)
                    else:
                        with open(featurefile, 'wb') as f:
                            np.save(f, x)

                    if feature not in partmeta['unordered_features']:
                        partmeta['unordered_features'].append(feature)

                    partmeta['bytesize'] += sys.getsizeof(x)

            for label in label_names:
                labelfile = '%s/Label-%s' % (partdir, DataSet.fname(label))

                x = tensorgetter(samples, label)
                x[np.isnan(x)] = BaseFeature.MISSING_VALUE

                if label in persisters:
                    persisters[label].save(labelfile, x)
                else:
                    with open(labelfile, 'wb') as f:
                        np.save(f, x)

                if numclasses > 0 and len(label_names) == 1 and 'class_counts' in datasetmeta:
                    if len(x.shape) == 1:
                        for cls in xrange(0, numclasses):
                            klass = 'Class_' + str(cls)
                            datasetmeta['class_counts'][klass] += sum(1 for y in x if y == cls)
                    elif len(x.shape) == 2 and x.shape[1] == 1:
                        for cls in xrange(0, numclasses):
                            klass = 'Class_' + str(cls)
                            datasetmeta['class_counts'][klass] += sum(1 for y in x if y[0] == cls)
                    else:
                        for cls in xrange(0, numclasses):
                            klass = 'Class_' + str(cls)
                            datasetmeta['class_counts'][klass] += sum(x[:, cls])

                partmeta['bytesize'] += sys.getsizeof(x)

            if not os.path.isfile(partdir + '/ids.txt'):
                with open(partdir + '/ids.txt', 'wb') as f:
                    f.writelines([x + "\n" for x in ids])

            with open(partdir + '/part.json', 'w') as f:
                json.dump(partmeta, f)
            logging.info('%s stored or updated. In total %d rows generated' % (partdir, totalrows))
            with open('input/%s/dataset_V5.json' % experiment_name, 'w') as f:
                json.dump(datasetmeta, f)
        sollfeatures = set([x['name'] for x in datasetmeta['features']])
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
                            x = input(
                                'Press y to remove the part, any other key to leave it (in this case missing feature will always have missing values)')
                            if x == 'y':
                                shutil.rmtree('input/%s/%s' % (experiment_name, entry))
        with open('input/%s/dataset_V5.json' % experiment_name, 'w') as f:
            json.dump(datasetmeta, f)
        for ff in datasetmeta['features']:
            for v in ff['top10values']:
                if len(v) == 0:
                    logging.warning('Feature %s has no values' % ff['name'])
                elif len(v) == 1:
                    if v.most_common(1)[0][0] == BaseFeature.MISSING_VALUE:
                        logging.warning('Feature %s has only missing values' % ff['name'])
                    else:
                        logging.warning('Feature %s has only one value %s' % (ff['name'], v.most_common(1)[0][0]))
                elif v.most_common(1)[0][1] > 0.99 * totalrows:
                    logging.warning('Feature %s has the value %s in more than 99%% of samples' % (ff['name'],
                                                                                                  v.most_common(1)[0][
                                                                                                      0]))
        if 'class_counts' in datasetmeta:
            notpresent = []
            lessthancent = []
            for k, v in datasetmeta['class_counts'].iteritems():
                if v == 0:
                    notpresent.append(str(k))
                if v < 0.01 * totalrows:
                    lessthancent.append(str(k))

            if len(notpresent) > 0 or len(lessthancent) > 0:
                raise Exception('There is a class distribution problem. Following classes '
                                'are not present in the dataset: %s. Following classes '
                                'contribute to less than 1%% of dataset: %s'
                                % (','.join(notpresent), ','.join(lessthancent)))
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


    def __init__(self, experiment_name, maxRAM = None, maxParts = None):
        self.experiment_name = experiment_name

        self.meta = None

        if os.path.isfile('input/%s/dataset_V5.json' % experiment_name):
            with open('input/%s/dataset_V5.json' % experiment_name, 'r') as f:
                self.meta = json.load(f)
        else:
            raise Exception('input/%s/dataset_V5.json not found, please enter right path or '
                            'generate the dataset first' % experiment_name)

        logging.info('Loading dataset %s' % experiment_name)

        self.average_part_size = 0
        self.numsamples = 0
        self.partmap_indexes = [0]
        self.partmap_names = [None]
        self.performance = {'totalbytesloaded': 0}
        self.model_input_shape = tuple(self.meta['model_input_shape'])
        self.label_shape = tuple(self.meta['label_shape'])
        self.strsampletype = self.meta['sampletype']

        self.metaparts = {}
        self._load_metaparts('input/%s' % experiment_name)
        if len(self.metaparts):
            self.average_part_size = self.average_part_size * 1.0 / len(self.metaparts)
        logging.info('Average part size %f' % self.average_part_size)

        if maxParts is None:
            if maxRAM is None:
                maxRAM = int(0.5 * os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))

            if self.average_part_size > 0:
                maxParts = maxRAM / self.average_part_size
            else:
                maxParts = len(self.metaparts)

            maxParts = int(maxParts)
            if maxParts == 0:
                maxParts = 1

        if maxParts > 0:
            logging.info('Creating dataset cache storing at most %d parts' % maxParts)
            self.cache = LRU(int(maxParts))
        else:
            self.cache = None

    def _load_metaparts(self, dir):
        for entry in scandir(dir):
            if entry.is_dir():
                if entry.name.startswith('part_'):
                    try:
                        with open('%s/part.json' % entry.path, 'r') as f:
                            partmeta = json.load(f)
                            self.metaparts[entry.path] = partmeta
                            self.numsamples += partmeta['numsamples']
                            self.partmap_indexes.append(self.numsamples)
                            self.average_part_size += partmeta['bytesize']
                            self.partmap_names.append(entry.path)
                    except:
                        logging.error('Cannot load part %s' % entry.path )
                else:
                    self._load_metaparts(entry.path)

    def _load_x(self, partdir, name, islabel):
        if islabel:
            f_file = partdir + 'Label-' + DataSet.fname(name)
        else:
            f_file = partdir + DataSet.fname(name)

        x = None
        if os.path.isfile(f_file+'.persister'):
            with open(f_file+'.persister', 'rb') as pf:
                persister = cPickle.load(pf)
            x = persister.load(f_file)
        elif os.path.isfile(f_file):
            x = np.load(f_file)
        else:
            logging.warning('%s does not contain feature %s, replacing with missing value' % (partdir, f_file))

        return x


    def _loadpart(self, partname):
        self.performance['totalbytesloaded'] += self.metaparts[partname]['bytesize']
        partdir = partname + '/'

        feat = {}
        for ff in self.meta['features']:
            x = self._load_x(partdir, ff['name'], False)
            if x is None:
                if Settings.OnFeatureMissing == 'raise':
                    raise Exception(
                        'Feature file %s is missing in the part %s. If you want its values to be replaced with MISSING_VALUE, set Settings.OnFeatureMissing to impute' % (ff['name'], partdir))
                else:
                    logging.info('Feature file %s is missing in the part %s.' % (ff['name'], partdir))
                    x = np.full((self.metaparts[partname]['numsamples'],)+tuple(ff['output_shape']), BaseFeature.MISSING_VALUE, dtype=ff['dtype'])
            feat[ff['name']] = x

        labe = {}
        for ff in self.meta['labels']:
            labe[ff['name']] = self._load_x(partdir, ff['name'], True)

        return (feat, labe)

    def _get_part_and_offset(self, sample_index):
        i = bisect_right(self.partmap_indexes, sample_index)
        if i > 0:
            offset = sample_index - self.partmap_indexes[i - 1]
            part, offset = self.partmap_names[i], offset
            return part, offset
        raise ValueError()

    def get_sample_id(self, sample_index):
        partname, offset = self._get_part_and_offset(sample_index)
        with open(partname+'/ids.txt', 'r') as f:
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
        if len(self.meta['labels']) > 1:
            raise Exception('Use get_one_sample instead of getitem, because you have to pass the label')

        featuremetas, labelmeta, x_shape, y_shape = self._getmetas()
        if x_shape != self.model_input_shape:
            raise Exception('Features of the dataset have shape %s, but model expect input shape %s' % (x_shape, self.model_input_shape))

        if y_shape is not None and y_shape != self.label_shape:
            raise Exception('Labels of the dataset have shape %s, but model delivers output of shape %s' % (y_shape, self.label_shape))

        return self.get_one_sample(index, featuremetas, labelmeta, x_shape)

    def get_one_sample(self, index, featuremetas, labelmeta, x_shape):
        if index >= self.numsamples:
            raise IndexError()

        partname, offset = self._get_part_and_offset(index)
        if self.cache is None:
            feat, labe = self._loadpart(partname)
        else:
            if partname not in self.cache:
                self.cache[partname] = self._loadpart(partname)

            feat,labe = self.cache[partname]

        x = combine_tensors([feat[ff['name']][offset:offset+1] for ff in featuremetas], x_shape, 1)[0]
        if labelmeta:
            y = labe[labelmeta['name']][offset]

        if labelmeta:
            return x, y
        else:
            return x, None

    def __len__(self):
        return self.numsamples

    def _getmetas(self, features=None, label=None):
        if features is None:
            featuremetas = self.meta['features']
        else:
            featuremetas = [ff for ff in self.meta['features'] if ff['name'] in features]
            if len(featuremetas) < len(features):
                raise Exception('Dataset %s does not contain the following features: %s' % (self.experiment_name, ','.join(set(features).difference(set([ff['name'] for ff in featuremetas])))))

        if label is None:
            if len(self.meta['labels']) > 1:
                raise Exception('Specify which label should be retrieved from the dataset')
            if len(self.meta['labels']) > 0:
                labelmeta = self.meta['labels'][0]
            else:
                labelmeta = None
        else:
            labelmeta = [ff for ff in self.meta['labels'] if ff['name'] == label]
            if len(labelmeta) == 0:
                raise Exception('Dataset does not contain label %s' % label)
            labelmeta=labelmeta[0]

        x_shape = BaseModel.calculate_shape(shapes=[ff['output_shape'] for ff in featuremetas])
        y_shape = tuple(labelmeta['output_shape']) if labelmeta is not None else None

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
            raise Exception('Features of the dataset have shape %s, but model expect input shape %s' % (x_shape, self.model_input_shape))

        if len(self.meta['labels']) == 1 and y_shape != self.label_shape:
            raise Exception('Label of the dataset has shape %s, but model delivers output of shape %s' % (y_shape, self.label_shape))

        all_x = np.zeros((self.numsamples,) + x_shape)
        if labelmeta:
            all_y = np.zeros((self.numsamples, )+ y_shape)

        row = 0
        for partname in self.partmap_names[1:]:
            feat, labe = self._loadpart(partname)
            numsamples = self.metaparts[partname]['numsamples']
            if self.cache:
                self.cache[partname] = (feat, labe)
            all_x[row:row+numsamples] = combine_tensors([feat[ff['name']] for ff in featuremetas], x_shape, numsamples)
            if labelmeta:
                all_y[row:row+numsamples] = labe[labelmeta['name']]
            row += numsamples

        if labelmeta:
            return all_x, all_y
        else:
            return all_x, None

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
        featmap = []
        for f in self.meta['features']:
            numcol = 1
            if len(f['output_shape']) > 0:
                numcol = f['output_shape'][0]
            for num in xrange(0, numcol):
                if human_readable:
                    featmap.append(f['name'] + "_" + str(num))
                else:
                    featmap.append(f['name'])
        return featmap

    def clone(self, experiment_name, index_array=None, processor=None, persisters={}):
        if index_array is None:
            index_array = np.arange(0, len(self))
        if processor is None:
            processor = lambda feature_name, X: X

        idx_iter = index_array.__iter__()

        part_size = self.metaparts.values()[0]['numsamples']
        nesting = DataSet._get_optimal_nesting(len(index_array) / part_size)
        feature_names = [x['name'] for x in self.meta['features']]
        label_names = [x['name'] for x in self.meta['labels']]

        class TensorCarrier(BaseSample):
            def __init__(self, entityid, tensors):
                BaseSample.__init__(self, entityid)
                self.tensors = tensors

        def clonesamplefetcher():
            idx = idx_iter.next()
            tensors = {}
            for f in feature_names:
                featuremetas, _, _, _= self._getmetas(features=[f])
                X, _ = self.get_one_sample(idx, featuremetas, None, tuple(featuremetas[0]['output_shape']))
                tensors[f] = processor(f, X)
            for f in label_names:
                _, labelmeta, _, _ = self._getmetas(label=f)
                _, y = self.get_one_sample(idx, [], labelmeta, ())
                tensors[f] = processor(f, y)
            sampleid = self.get_sample_id(idx)
            return TensorCarrier(sampleid, tensors)

        def clonetensorgetter(samples, featurename):
            return np.stack([x.tensors[featurename] for x in samples])

        newmeta = self.meta.copy()
        for ff in newmeta['features']:
            if len(ff['output_shape']) == 1:
                numtop = ff['output_shape'][0]
            else:
                numtop = 1
            ff['top10values'] = [Counter() for _ in xrange(0, numtop)]
        newmeta['class_counts'] = {}

        DataSet._write(
            newmeta,
            experiment_name,
            feature_names,
            label_names,
            nesting,
            self.meta['numclasses'],
            part_size,
            clonesamplefetcher,
            clonetensorgetter,
            persisters
        )



    def plot_data(self, bins=10, all_in_one=True, index_array=None):
        if self.meta['numclasses'] > 1:
            self.plot_data_classification(bins, all_in_one, index_array)
        else:
            self.plot_data_regression(all_in_one, index_array)

    def plot_data_classification(self, bins=10, all_in_one=True, index_array = None):
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
            numclasses = self.meta['numclasses']
            for class_index in xrange(0, numclasses):
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

    def plot_data_regression(self, all_in_one=True, index_array = None):
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
        for i,s in enumerate(x):
            img = Image.frombytes(self.mode, (self.width, self.height), s)
            with open('%s_%i.jpg' % (featurefile, i), 'wb') as f:
                img.save(f, format='JPEG')
        self.numimages = len(x)
        with open('%s.persister' % (featurefile), 'wb') as f:
            cPickle.dump(self, f)

    def load(self, featurefile):
        x = np.full((self.numimages, self.width, self.height, self.numchannels), BaseFeature.MISSING_VALUE)
        for i in xrange(0, self.numimages):
            file = '%s_%i.jpg' % (featurefile, i)
            if os.path.isfile(file):
                img = Image.open(file)
                x[i] = np.asarray(img, dtype=np.uint8)
        return x


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    #print DataSet._get_optimal_nesting(10)
    #print DataSet._get_optimal_nesting(1000)
    #print DataSet._get_optimal_nesting(50000)
    print DataSet._get_optimal_nesting(400000)
    #print DataSet._get_optimal_nesting(1000000)