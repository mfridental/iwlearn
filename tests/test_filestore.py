# -*- coding: utf-8 -*-
import logging
import sys
import datetime as dt
from pyclickhouse import Connection
import pytest
from sklearn.model_selection import train_test_split

from iwlearn.models import ScikitLearnModel
from iwlearn.training import DataSet
from iwlearn.storage import FileStore
from common import TestOfferSample, OneHotFeature
import shutil
import time
# with capsys.disabled():
#     metrics, _, _ = model.evaluate_and_print(test)

class TestFixture(object):
    def setup_class(cls):
        try:
            shutil.rmtree('input/filestore')
        except Exception as e:
            pass

        cls.Samples = []
        cls.SomeDate = dt.datetime.now()
        for i in range(1000):
            sample = TestOfferSample(str(i))
            sample['an_int'] = 42
            sample['a_float'] = 3.1415
            sample['a_string'] = 'hello'
            sample['a_date'] = cls.SomeDate.date()
            sample['a_datetime'] = cls.SomeDate
            sample['a_bool'] = True
            sample['a_list'] = [42, 3.1415, 'hello', cls.SomeDate.date(), cls.SomeDate, True]
            cls.Samples.append(sample)

    def test_insert(self, capsys):
        fs = FileStore(TestOfferSample)

        fs.insert_samples(self.Samples)
        samples = fs.find_samples()
        assert len(samples) == len(self.Samples)
        sdict = dict()
        for sample in samples:
            sdict[sample.entityid] = sample

        for orig in self.Samples:
            restored = sdict[orig.entityid]
            for aval, bval in zip(orig.data.items(), restored.data.items()):
                assert aval[1] == bval[1]

    def test_find_filtering(self, capsys):
        try:
            shutil.rmtree('input/filestore')
        except Exception as e:
            pass

        fs = FileStore(TestOfferSample)

        s = self.Samples[0]
        fs.insert_sample(s)
        s['created'] = s['created'] - dt.timedelta(days=1)
        fs.insert_sample(s)
        samples = fs.find_samples()
        assert len(samples) == 2

        s2 = self.Samples[1]
        fs.insert_sample(s2)
        s2['created'] = s2['created'] - dt.timedelta(minutes=5)
        fs.insert_sample(s2)
        samples = fs.find_samples()
        assert len(samples) == 4

        # only 0th
        samples = fs.find_samples(match_entityid=lambda x: x == s.entityid)
        assert len(samples) == 2
        assert samples[0].entityid == s.entityid
        assert samples[1].entityid == s.entityid

        # only 1st
        samples = fs.find_samples(match_entityid=lambda x: x == s2.entityid)
        assert len(samples) == 2
        assert samples[0].entityid == s2.entityid
        assert samples[1].entityid == s2.entityid

        # only today
        samples = fs.find_samples(earliest=dt.date.today())
        assert len(samples) == 3

        # only yesterday
        samples = fs.find_samples(latest=dt.date.today()-dt.timedelta(days=1))
        assert len(samples) == 1

        # only one
        samples = fs.find_samples(earliest=s2['created'], latest=s2['created'])
        assert len(samples) == 1

        sample = fs.find_latest_sample(s2.entityid)
        assert sample['created'] == s2['created'] + dt.timedelta(minutes=5)

    def test_replace(self, capsys):
        try:
            shutil.rmtree('input/filestore')
        except Exception as e:
            pass

        fs = FileStore(TestOfferSample)

        s = self.Samples[0]
        fs.insert_sample(s)
        failed = False
        try:
            fs.insert_sample(s)
        except:
            failed = True
        assert failed
        orig = fs.find_latest_sample(s.entityid)
        s['a_float'] = 2.73
        fs.replace_sample(s)
        newone = fs.find_latest_sample(s.entityid)
        assert orig['a_float'] == 3.1415
        assert newone['a_float'] == 2.73

if __name__ == "__main__":
    pytest.main([__file__])
