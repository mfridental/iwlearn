# TODO: KERAS MODELS ARE TEMPORARY BROKEN DUE TO TENSORFLOW API CHANGE


# # -*- coding: utf-8 -*-
# import sys
# import logging
# import dill
# import ujson as json
# import os
# import datetime as dt
#
# import numpy as np
# import tensorflow.compat.v1.keras as k
# from keras.preprocessing.image import Iterator
# import tensorflow as tf
#
# from sklearn.metrics import accuracy_score, \
#     mean_absolute_error  # keras metrics do not deliver a single value, but a tensor - not suitable for
# # BaseModel.evaluate
#
# from iwlearn.base import BaseModel, ScorePrediction, TrivialFeature
#
#
# class KerasClassifierPredictionFactory(object):
#     """
#     Needed to create KerasClassifierPredictions
#     """
#
#     def create(self, scores, predictor):
#         return KerasClassifierPrediction(scores, predictor)
#
#
# class KerasClassifierPrediction(ScorePrediction):
#     """
#     We need to create a special prediction because Keras classifiers use softmax output with shape (numclasses,),
#     while normal classifiers have output shape () - scalar
#     """
#
#     def __init__(self, scores, predictor):
#         ScorePrediction.__init__(self, scores, predictor)
#         if self.scores is not None and 0 < len(self.scores):
#             self.prediction = np.zeros_like(self.scores)
#             self.prediction[np.argmax(self.scores)] = 1
#
#
# class BaseKerasModel(BaseModel):
#     def __init__(self, name, features, sampletype, predictionfactory=None, labelkey=None):
#         BaseModel.__init__(self, name, features, sampletype, predictionfactory, labelkey)
#         self.experiment_name = self.__class__.__name__
#         self.custom_objects = {}
#         self.kerasmodel = None
#         self.kerasmodelfile = None
#
#     @classmethod
#     def load(cls, filepath):
#         result = BaseModel.load(filepath)
#         if result.kerasmodelfile is not None:
#             result.kerasmodel = k.models.load_model(result.kerasmodelfile, compile=True,
#                                                     custom_objects=result.custom_objects)
#         return result
#
#     def train_impl(self, dataset, **configuration):
#         self._createkerasmodel()
#
#         logging.info('Run training loop')
#
#         batch_size = 32
#         num_epochs = 100
#         image_generator = None
#         shuffle = True
#         seed = dt.datetime.now().microsecond
#         callbacks = [
#             k.callbacks.TensorBoard(log_dir='logs/%s' % self.experiment_name,
#                                     histogram_freq=0,
#                                     batch_size=32,
#                                     write_graph=True,
#                                     write_grads=False,
#                                     write_images=False,
#                                     embeddings_freq=0,
#                                     embeddings_layer_names=None,
#                                     embeddings_metadata=None),
#             k.callbacks.EarlyStopping(
#                 monitor='categorical_accuracy',
#                 min_delta=0.01,
#                 patience=3,
#                 verbose=1)]
#
#         if 'batch_size' in configuration:
#             batch_size = configuration['batch_size']
#
#         if 'num_epochs' in configuration:
#             num_epochs = configuration['num_epochs']
#
#         if 'callbacks' in configuration:
#             callbacks = configuration['callbacks']
#
#         if 'image_generator' in configuration:
#             image_generator = configuration['image_generator']
#
#         if 'shuffle' in configuration:
#             shuffle = configuration['shuffle']
#
#         if 'seed' in configuration:
#             seed = configuration['seed']
#
#         sess = tf.compat.v1.Session()
#
#         with sess.as_default():
#             k.backend.set_session(sess)
#
#             init_op = tf.global_variables_initializer()
#             sess.run(init_op)
#
#             self.kerasmodel.fit_generator(KerasGenerator(dataset, batch_size, image_generator, shuffle, seed),
#                                           verbose=1,
#                                           epochs=num_epochs,
#                                           callbacks=callbacks)
#
#     def _predict_scores_impl(self, X):
#         return self.kerasmodel.predict(X, verbose=1)
#
#     def _save(self, folder, creator):
#         filepath_modelname = '%s/%s' % (folder, self.tag,)
#
#         if not os.path.exists(filepath_modelname):
#             os.makedirs(filepath_modelname)
#
#         filepath_model = filepath_modelname + '.h5'
#         self.kerasmodel.save(filepath_model)
#         self.kerasmodelfile = filepath_model
#
#         self.meta_infos["filepath_model"] = filepath_model
#
#         BaseModel._save(self, folder, creator)
#
#         return filepath_modelname
#
#     def _createkerasmodel(self):
#         raise NotImplementedError
#
#
# class KerasClassifierLabel(TrivialFeature):
#     def __init__(self, path, output_shape):
#         TrivialFeature.__init__(self, path, output_shape)
#
#     def get(self, sample):
#         y = TrivialFeature.get(self, sample)
#         if self.output_shape == ():
#             return k.utils.to_categorical(y, self._get_width())
#         return y
#
#
# class BaseKerasClassifierModel(BaseKerasModel):
#     def __init__(self, name, features, sampletype, predictionfactory=None,
#                  labelkey=None):
#         if predictionfactory is None:
#             predictionfactory = KerasClassifierPredictionFactory()
#         BaseKerasModel.__init__(self, name, features, sampletype, predictionfactory, '')
#         self.output_shape = (numclasses,)
#         self.metrics = [accuracy_score]
#         if labelkey == '':
#             self.labels = []
#         elif labelkey is None:
#             self.labels = [KerasClassifierLabel(self.task + 'Label', self.output_shape)]
#         else:
#             self.labels = [KerasClassifierLabel(labelkey, self.output_shape)]
#
#
# class BaseKerasRegressorModel(BaseKerasModel):
#     def __init__(self, name, features, sampletype, numoutputs, prediction_factory=None, labelkey=None):
#         BaseModel.__init__(self, name, features, sampletype, prediction_factory, labelkey)
#         self.output_shape = (numoutputs,)
#         self.metrics = [mean_absolute_error]
#
#
# class KerasGenerator(Iterator):
#     def __init__(self, dataset, batch_size, image_generator, shuffle, seed):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.image_generator = image_generator
#         self.fourdshape = None
#         if self.image_generator is not None and len(self.dataset.sampleshape) > 4:
#             self.fourdshape = (int(np.prod(self.dataset.sampleshape[0:-3])),) + tuple(self.dataset.sampleshape[-3:])
#         Iterator.__init__(self, len(self.dataset), self.batch_size, shuffle, seed)
#
#     def _get_batches_of_transformed_samples(self, index_array):
#         x, y = self.dataset.get_samples(index_array)
#
#         if self.image_generator is not None:
#             if self.fourdshape is not None:
#                 x = np.reshape(x, self.fourdshape)
#             x = [self.image_generator.random_transform(imagetensor) for imagetensor in x]
#             x = [self.image_generator.standardize(imagetensor) for imagetensor in x]
#             if self.fourdshape is not None:
#                 x = np.reshape(x, self.dataset.sampleshape)
#
#         return x, y
#
#
# if __name__ == "__main__":
#     logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
