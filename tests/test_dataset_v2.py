# -*- coding: utf-8 -*-
import sys
import logging
import datetime as dt
import uuid
import pytest
import numpy as np

from tests.common import TestOfferSample, SimpleSampleFeature
from iwlearn.training import DataSet
from iwlearn.training.dataset_operations import _generate_folds
from iwlearn.models import ScikitLearnModel
from iwlearn import BaseFeature


class TestFixture():

    def test_expansion(self, capsys, caplog, monkeypatch):
        with capsys.disabled():
            import shutil
            try:
                shutil.rmtree('input/v2_test')
            except:
                pass

        samples = []
        samples_2 = []
        for x in range(0, 100):
            samples.append(TestOfferSample.fromjson(
                {
                    'entityid': uuid.uuid4(),
                    'value_1': x,
                    'value_2': x % 17,
                    'TestModelLabel': (1 if x % 17 == 0 else 0)
                }))
            samples_2.append(TestOfferSample.fromjson(
                {
                    'entityid': uuid.uuid4(),
                    'value_1': 2 * x,
                    'value_2': 2 * x % 17,
                    'TestModelLabel': 1 if x % 17 == 0 else 0
                }))

        model_1 = ScikitLearnModel('TestModel', [SimpleSampleFeature('value_1')], TestOfferSample)
        DataSet.bootstrap('v2_test', model_1, samples, part_size=10)
        ds = DataSet('v2_test')

        assert len(ds) == 100
        assert tuple(ds.meta.model_input_shape) == (1,)
        assert [x.name for x in ds.meta.features] == ['SimpleSampleFeature_value_1']
        assert len(ds.meta.features) == 1
        assert len(ds.metaparts) == 10
        for k, p in ds.metaparts.items():
            assert p['unordered_features'] == ['SimpleSampleFeature_value_1']

        # Expand vertically
        DataSet.bootstrap('v2_test', model_1, samples_2, part_size=10)
        ds = DataSet('v2_test')

        assert len(ds) == 200
        assert tuple(ds.meta.model_input_shape) == (1,)
        assert [x.name for x in ds.meta.features] == ['SimpleSampleFeature_value_1']
        assert len(ds.meta.features) == 1
        assert len(ds.metaparts) == 20
        for k, p in ds.metaparts.items():
            assert p['unordered_features'] == ['SimpleSampleFeature_value_1']

        # Expand horizontally
        model_2 = ScikitLearnModel('TestModel', [SimpleSampleFeature('value_1'), SimpleSampleFeature('value_2')],
                                   TestOfferSample)
        caplog.clear()
        DataSet._test_global_remove_parts_setting = 'n'  # do not remove parts with missing feature
        DataSet.bootstrap('v2_test', model_2, samples, part_size=10)
        ds = DataSet('v2_test')
        assert len(caplog.records) == 11
        assert caplog.records[0].message == 'Model input shape has been changed from (1,) to (2,)'
        for tpl in caplog.records[1:]:
            assert "does not contain following features: {'SimpleSampleFeature_value_2'}" in tpl.msg
        assert len(ds) == 200
        assert tuple(ds.meta.model_input_shape) == (2,)
        assert [x.name for x in ds.meta.features] == ['SimpleSampleFeature_value_1',
                                                            'SimpleSampleFeature_value_2']
        assert len(ds.metaparts) == 20
        for k, p in ds.metaparts.items():
            assert 'SimpleSampleFeature_value_1' in p['unordered_features']
            assert len(p['unordered_features']) == 1 or len(p['unordered_features']) == 2

        # Check dataset would not crash for parts do not containing the second feature
        X, y_true = ds.get_all_samples()
        assert X.shape == (200, 2)
        import numpy as np
        assert np.mean(X[:, 0]) != BaseFeature.MISSING_VALUE
        assert np.mean(X[:, 1]) != BaseFeature.MISSING_VALUE
        assert sum(X[:, 1] == BaseFeature.MISSING_VALUE) == 100

        # Remove first feature
        caplog.clear()
        DataSet.bootstrap('v2_test', model_1, samples, part_size=10)
        ds = DataSet('v2_test')
        assert len(caplog.records) == 2
        assert caplog.records[1].message == 'Model input shape has been changed from (2,) to (1,)'
        assert "Following features removed from dataset_V8.json: {'SimpleSampleFeature_value_2'}" in \
               caplog.text

    def test_generate_folds(self):
        y = [0]*10 + [1]* 10

        test, train = _generate_folds(y, fold=5, numclasses=2)
        assert len(train) == 5
        assert len(test) == 5
        assert len(test[0]) == 4
        assert len(train[0]) == 16
        tmp = set()
        for t in train:
            tmp.update(t)
        assert len(tmp) == 20
        tmp = set()
        for t in test:
            tmp.update(t)
        assert len(tmp) == 20

        y = np.asarray(y)
        test, train = _generate_folds(y,
                                      fold=5,
                                      numclasses=2,
                                      train_class_sizes=[0.5,1],
                                      test_class_sizes=None)
        assert len(train[0]) == 12
        assert sum(y[train[0]]==0) == 4
        assert sum(y[train[1]]==1) == 8
        assert len(test[0]) == 4
        assert sum(y[test[0]]==0) == 2
        assert sum(y[test[1]]==1) == 2

        A = 24000
        B = 800
        y = [0]*A + [1]* B
        y = np.asarray(y)
        test, train = _generate_folds(y,
                                      fold=5,
                                      numclasses=2,
                                      train_class_sizes=[1.5,1],
                                      test_class_sizes=None)
        assert len(train[0]) == 1600 # 960+640
        assert sum(y[train[0]]==0) == 960 # 640 * 1.5
        assert sum(y[train[1]]==1) == 640 # 800-160
        assert len(test[0]) == 4960 #(A+B)/5
        assert sum(y[test[0]]==0) == 4800 # 4960 - 160
        assert sum(y[test[1]]==1) == 160 # B/(A+B)*4960



if __name__ == "__main__":
    pytest.main([__file__])
