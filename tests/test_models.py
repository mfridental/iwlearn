# -*- coding: utf-8 -*-
import logging
import sys
from pyclickhouse import Connection
import pytest
from sklearn.model_selection import  train_test_split

from iwlearn.models import ScikitLearnModel
from iwlearn.training import DataSet
from common import TestOfferSample, OneHotFeature

class TestFixture(object):
    def setup_class(cls):
        model = ScikitLearnModel('TestModel', [OneHotFeature()], TestOfferSample)

        train_samples = [
            TestOfferSample.fromjson({'entityid': str(i), 'TestModelLabel': i % 2}) for i in
            xrange(0, 10000)]
        test_samples = [
            TestOfferSample.fromjson({'entityid': str(i), 'TestModelLabel': i % 2}) for i in
            xrange(10000, 12000)]

        DataSet.remove('bootstrapped_training')
        DataSet.bootstrap('bootstrapped_training', model, train_samples, part_size=2000, numclasses=2)
        DataSet.remove('bootstrapped_test')
        DataSet.bootstrap('bootstrapped_test', model, test_samples, part_size=2000, numclasses=2)

    def test_training(self, capsys):
        model = ScikitLearnModel('TestModel', [OneHotFeature()], TestOfferSample)

        train = DataSet('bootstrapped_training')
        test = DataSet('bootstrapped_test')
        print 'Training'
        model.train(train)
        with capsys.disabled():
            metrics, _, _ = model.evaluate_and_print(test)
        assert 'f1_score' in metrics
        assert 'precision_score' in metrics
        assert 'recall_score' in metrics
        assert metrics['f1_score'] > 0.1


if __name__ == "__main__":
    pytest.main([__file__])
