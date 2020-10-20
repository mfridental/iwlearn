# # -*- coding: utf-8 -*-
# from pyclickhouse import Connection
# import pytest
#
# from iwlearn.training import DataSet
# from common import TestOfferSample, SimpleKerasModel, OneHotFeature
# from iwlearn.mongo import setmongouri, mongoclient, insert_samples
# from iwlearn.models.based_on_keras import KerasGenerator
#
#
# class TestFixture(object):
#     def setup_class(cls):
#         model = SimpleKerasModel('TestModel', [OneHotFeature()], TestOfferSample)
#
#         train_samples = [TestOfferSample.fromjson({'entityid': str(i), 'TestModelLabel': [0, 1] if i % 2 == 0 else [1, 0]}) for i in range(0, 10000)]
#         test_samples = [TestOfferSample.fromjson({'entityid': str(i), 'TestModelLabel': [0, 1] if i % 2 == 0 else [1, 0]}) for i in range(10000, 12000)]
#
#         DataSet.remove('batching_train')
#         DataSet.bootstrap('batching_train',model, train_samples, part_size=2000)
#         DataSet.remove('batching_test')
#         DataSet.bootstrap('batching_test',model, test_samples, part_size=2000)
#
#     def test_kerasgenerator(self, capsys):
#         with capsys.disabled():
#             ds = DataSet('batching_train', maxRAM=13302*1000)
#             print (ds.meta)
#             gen = KerasGenerator(ds, 10, None, True, 1)
#             x, y = gen._get_batches_of_transformed_samples([0,3,14])
#             assert x.shape == (3,) + OneHotFeature().output_shape
#             assert y.shape == (3,2)
#             ds2 = DataSet('batching_test', maxRAM=13302*1000)
#             assert len(ds)+len(ds2) >= 10000
#             assert len(ds.metaparts) >= 4
#
#     def test_simplekeras(self, capsys):
#         model = SimpleKerasModel('TestModel', [OneHotFeature()], TestOfferSample)
#         train = DataSet('batching_train', maxRAM=13302*1000)
#         with capsys.disabled():
#             model.train(train, batch_size = 1000)
#             test = DataSet('batching_test', maxRAM=13302 * 1000)
#             print (len(test))
#             print (model.evaluate(test))
#
# if __name__ == "__main__":
#     pytest.main([__file__])
