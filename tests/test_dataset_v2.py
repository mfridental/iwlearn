# -*- coding: utf-8 -*-
import sys
import logging
import datetime as dt
import uuid
import pytest

from tests.common import TestOfferSample, SimpleSampleFeature
from iwlearn.training import DataSet
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
        DataSet.bootstrap('v2_test', model_1, samples, part_size=10, numclasses=2)
        ds = DataSet('v2_test')

        assert len(ds) == 100
        assert tuple(ds.meta['model_input_shape']) == (1,)
        assert [x['name'] for x in ds.meta['features']] == ['SimpleSampleFeature_value_1']
        assert len(ds.meta['features']) == 1
        assert len(ds.metaparts) == 10
        for k, p in ds.metaparts.items():
            assert p['unordered_features'] == ['SimpleSampleFeature_value_1']

        # Expand vertically
        DataSet.bootstrap('v2_test', model_1, samples_2, part_size=10, numclasses=2)
        ds = DataSet('v2_test')

        assert len(ds) == 200
        assert tuple(ds.meta['model_input_shape']) == (1,)
        assert [x['name'] for x in ds.meta['features']] == ['SimpleSampleFeature_value_1']
        assert len(ds.meta['features']) == 1
        assert len(ds.metaparts) == 20
        for k, p in ds.metaparts.items():
            assert p['unordered_features'] == ['SimpleSampleFeature_value_1']

        # Expand horizontally
        model_2 = ScikitLearnModel('TestModel', [SimpleSampleFeature('value_1'), SimpleSampleFeature('value_2')],
                                   TestOfferSample)
        caplog.clear()
        DataSet._test_global_remove_parts_setting = 'n'  # do not remove parts with missing feature
        DataSet.bootstrap('v2_test', model_2, samples, part_size=10, numclasses=2)
        ds = DataSet('v2_test')
        assert len(caplog.records) == 10
        for tpl in caplog.records:
            assert "does not contain following features: {'SimpleSampleFeature_value_2'}" in tpl.msg
        assert len(ds) == 200
        assert tuple(ds.meta['model_input_shape']) == (2,)
        assert [x['name'] for x in ds.meta['features']] == ['SimpleSampleFeature_value_1',
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
        DataSet.bootstrap('v2_test', model_1, samples, part_size=10, numclasses=2)
        ds = DataSet('v2_test')
        assert len(caplog.records) == 1
        assert "Following features removed from dataset_V5.json: {'SimpleSampleFeature_value_2'}" in \
               caplog.text


if __name__ == "__main__":
    pytest.main([__file__])
