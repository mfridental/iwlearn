# -*- coding: utf-8 -*-
import pytest
import numpy as np

from iwlearn.base import BaseSample, BaseFeature, BaseModel
from iwlearn.training import DataSet

import shutil


class TestDataSet(object):

    def setup_class(cls):
        try:
            shutil.rmtree('input/testdataset')
        except:
            pass
        cls.samples = []
        for d in range(0, 1000):
            cls.samples.append({'entityid': str(d), '_id': str(d), 'value': d, 'TestModelLabel': d % 2})

        class SimpleFeature(BaseFeature):
            def __init__(self):
                BaseFeature.__init__(self)
                self.output_shape = (1,)
                self.name = 'SimpleFeature'

            def get(self, sample):
                return [sample.data['value']]

        class VectorFeature(BaseFeature):
            def __init__(self):
                BaseFeature.__init__(self)
                self.output_shape = (6,)

            def get(self, sample):
                return [sample.data['value'] * 10] * 6

        class LSTMFeature(BaseFeature):
            def __init__(self):
                BaseFeature.__init__(self)
                self.output_shape = (10, 200, 200, 3)

            def get(self, sample):
                return np.full((10, 200, 200, 3), sample.data['value'] * 20)

        class SimpleSample(BaseSample):
            pass

        cls.onesample = SimpleSample.fromjson({'entityid': 5, '_id': '5', 'value': 5})

        cls.threesamples = [
            SimpleSample.fromjson({'entityid': 1, 'value': 1, '_id': '1'}),
            SimpleSample.fromjson({'entityid': 2, 'value': 2, '_id': '2'}),
            SimpleSample.fromjson({'entityid': 3, 'value': 3, '_id': '3'})
        ]

        class TestDimensionsModel(BaseModel):
            def __init__(self):
                self.features = [SimpleFeature(), VectorFeature(), LSTMFeature()]
                self.sampletype = SimpleSample
                BaseModel.__init__(self, task='TestModel', features=self.features, sampletype=self.sampletype)

        cls.dimensionsmodel = TestDimensionsModel()

        class TestModel(BaseModel):
            def __init__(self):
                self.features = [SimpleFeature()]
                self.sampletype = SimpleSample
                BaseModel.__init__(self, task='TestModel', features=self.features, sampletype=self.sampletype,
                                   labelkey='TestModelLabel')

        cls.generatormodel = TestModel()

        class MockCollection():
            def find(self, query, batch_size, **kwargs):
                return cls.samples.__iter__()

        class MockMongoDb(dict):
            def __init__(self):
                self['SimpleSamples'] = MockCollection()

            def command(self, a, b):
                return {'avgObjSize': 1}

        class MockMongoClient(dict):
            def __init__(self):
                self['IWLearn'] = MockMongoDb()

            def close(self):
                pass

        cls.mongoclient = MockMongoClient()

    def teardown_class(cls):
        pass

    def testdimensions(self):
        x = self.dimensionsmodel.createX(self.threesamples)
        assert x.shape == (3, 17, 200, 200, 3)

    def testfeaturegeneration(self):
        x = self.dimensionsmodel.createX([self.onesample])
        assert x[0][0][0][0][0] == 5
        assert np.mean(x[0][0]) == 5.0
        assert x[0][1][0][0][0] == 50
        assert x[0][2][0][0][0] == 50
        assert x[0][3][0][0][0] == 50
        assert x[0][4][0][0][0] == 50
        assert x[0][5][0][0][0] == 50
        assert x[0][6][0][0][0] == 50
        assert np.mean(x[0][1]) == 50.0
        assert np.mean(x[0][2]) == 50.0
        assert np.mean(x[0][3]) == 50.0
        assert np.mean(x[0][4]) == 50.0
        assert np.mean(x[0][5]) == 50.0
        assert np.mean(x[0][6]) == 50.0
        assert x[0][7][0][0][0] == 100.0
        assert x[0][8][0][0][0] == 100.0
        assert np.mean(x[0][7]) == 100.0
        assert np.mean(x[0][8]) == 100.0

    def testgenerator(self):
        DataSet.generate('testdataset', self.generatormodel, maxRAM=288, customclient=self.mongoclient, query='query',
                          filter={})
        ds = DataSet('testdataset')
        print(len(ds))
        assert len(ds) == 1000
        x, y = ds[0]
        assert y == 0
        assert x.shape == (1,)

        x, y = ds[10]
        xx, yy = ds[9]
        assert x[0] - xx[0] == 1.0

        x, y = ds[150]
        assert y == 150 % 2

        assert len(ds.cache) == 2
        for i in range(0, 1000):
            assert ds[i][1] == i % 2

        X, y_true = ds.get_all_samples()
        assert X.shape == (1000, 1)
        assert np.sum(X) == np.sum(range(0, 1000))
        assert y_true.shape == (1000,)
        assert np.sum(y_true) == 0.5 * len(ds)

        X, y_true = ds.get_samples([14, 15])
        assert X.shape == (2, 1)
        assert X[1][0] - X[0][0] == 1
        assert y_true.shape == (2,)
        assert y_true[0] + y_true[1] == 1


if __name__ == "__main__":
    pytest.main([__file__])
