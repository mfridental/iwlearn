# -*- coding: utf-8 -*-
import datetime as dt
import inspect
import json
import logging
import os
import sys
from decimal import Decimal
import time

import dill
import numpy as np
from scandir import scandir
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix


class BaseOfBases(object):
    """
    Any entity handing or holding data should know its name, shape and dtype.
    """

    def __init__(self):
        self.name = self.__class__.__name__
        self.output_shape = ()
        self.dtype = np.dtype('object')

    def _get_width(self):
        return self.output_shape[0] if len(self.output_shape) > 0 else 1


class BaseSample(BaseOfBases):
    """
    Represents the raw information about a single sample from a dataset. Conceptually, you can think of the samples as
    normal dictionaries who know how to retrieve their data from the databases, using instances of BaseDataSource
    classes or other samples. From this raw information, the features and if needed labels can be calculated by
    instances of BaseFeature classes, and then persisted locally using the Dataset class. The samples themselves can
    be persisted in a database and re-used for training or evaluation.

    In the simple cases yuo should not inherit from the BaseSample. Instead, inherit from the database-specific samples
    defined in iwlearn.samples, as they provide functions retrieving data from some popular databases.

    When inheriting from this class, you have to define a constructor getting one parameter with the ID of the
    entity this sample is representing and passing it to the base constructor, eg
    ```python
    def __init__(self, userid):
        BaseSample.__init__(self, userid)
    ```
    and then the method makeimpl, defining how exactly the sample can retieve all needed data related to self.entityid.
    Note that you can reuse existing samples or datasources, eg
    ```python
    def makeimpl(self):
        useraccount = self.add(UserAccountDataSample(self.entityid))
        self.add(UserPaymentsDataSource(), self.entityid)
        self.add(PaymentStatisticDataSource(), useraccount['country'], useraccount['zip'])
    ```
    """

    def __init__(self, entityid):
        BaseOfBases.__init__(self)
        self.data = {'entityid': entityid, 'created': dt.datetime.now()}

    @classmethod
    def fromjson(cls, data):
        if 'entityid' not in data:
            raise Exception('Cannot deserialise %s: no entityid in the data' % cls.__name__)
        instance = cls(data['entityid'])
        instance.update(data)
        instance.checkdataconsistency()
        return instance

    def checkdataconsistency(self):
        if 'entityid' not in self.data:
            logging.error('Entityid is not present in sample %s' % self)
        if 'created' not in self.data:
            logging.error('Created is not present in sample %s' % self)

    def update(self, data):
        """
        Update the sample with additional data from the passed dictionary. This is not related to any persistance,
        it is really just extending the sample dict with the passed dict.
        """
        for key, value in data.items():
            self.data[key] = value

    def make(self):
        """
        Retrieve all data to create one fully defined sample containing all data needed for the model, related to
        the self.entityid.
        :return: Dictionary with the sample data.
        """
        started = dt.datetime.now()
        result = None
        try:
            result = self.makeimpl()
            if result is not None:
                if isinstance(result, dict):
                    self.data.update(result)
                else:
                    self.data = result
                logging.info('%s made in %.0f ms' % (self.name, (dt.datetime.now() - started).total_seconds() * 1000))
            else:
                logging.info('Cannot make %s ' % self.name)
        except:
            logging.exception('Cannot make sample %s ' % self)
        return result

    def add(self, source, **kwargs):
        """
        Make a sub-sample or a datasource and add its data to the current sample, under the key source.name with removed
        "Sample" or "DataSource" suffix. Raises an exception if the sub-sample or datasource cannot be made. If you
        don't want exception to be raised, use the tryadd method instead.

        When adding a datasource, pass named parameters to that data source, otherwise it would not know what data to
        retrieve. All named parameter will be passed to the makeimpl method of the data source.

        When adding a sub-sample, no need to add named parameter, because you will define the id of the sub-entity when
        instantiating the sub-sample, so it knows by itself, what data are to be retrieved.
        :param source: A sample or a data source
        :param kwargs: Named parameters for the data source
        :return: Dictionary with the data retrieved by the sample or data source.
        """
        result = self.tryadd(source, **kwargs)
        if result is None:
            raise Exception('Cannot add %r to %r' % (source, self))
        return result

    def tryadd(self, source, **kwargs):
        """
        Make a sub-sample or a datasource and add its data to the current sample, under the key source.name with removed
        "Sample" or "DataSource" suffix. If the sub-sample or datasource cannot be made, ignores it and returns None.
        If this information is vitally required for your further prediction or training process, use the method add
        instead, that would raise an Exception in case of missing data and therefore interrupt the processing workflow.

        When adding a datasource, pass named parameters to that data source, otherwise it would not know what data to
        retrieve. All named parameter will be passed to the makeimpl method of the data source.

        When adding a sub-sample, no need to add named parameter, because you will define the id of the sub-entity when
        instantiating the sub-sample, so it knows by itself, what data are to be retrieved.
        :param source: A sample or a data source
        :param kwargs: Named parameters for the data source
        :return: Dictionary with the data retrieved by the sample or data source, or None
        """
        key = source.name
        if key.endswith('Sample'):
            key = key[0: -len('Sample')]
        elif key.endswith('DataSource'):
            key = key[0: - len('DataSource')]

        if key not in self.data:
            result = source.make(**kwargs)
            if result is not None:
                self.data[key] = result
            return result

        return self.data[key]

    def makeimpl(self):
        """
        Override and return a dictionary with the raw data retrieved from the data source, or None if there is no
        possibility to retrieve.
        """
        raise NotImplementedError()

    def __str__(self):
        return self.entityid

    def __repr__(self):
        if '_id' in self.data:
            return '%s(%s,_id=%s)' % (self.name, str(self.entityid), repr(self.id))
        if 'entityid' in self.data:
            return '%s(%s)' % (self.name, str(self.entityid))
        return 'Unidentified Sample'

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, item):
        return item in self.data

    @property
    def id(self):
        return self.data['_id']

    @property
    def created(self):
        return self.data['created']

    @property
    def entityid(self):
        return self.data['entityid']


class BaseFeature(BaseOfBases):
    """
    The task of a feature is to calculate a float or int scalar, or a tensor, from the data stored in the sample. The
    resulting value can then be used to compose the input tensor of your rule or model, or to generate the labels.

    If the feature can be calculated simply by returning a numeric value stored somewhere in the sample, use the
    TrivialFeature instead.

    When inheriting:
        - decide, what output shape the feature will have. See documentation to obtain the guidelines for encoding of
        some complicated data like prices, geo coordinates, categorical data, etc.
        - overwrite constructor, call the base and set the output shape of that feature. Also, it is recommended to
        set the dtype of the feature for large output shapes - for example, if the feature is to return the pixels of
        an image, it is a good idea to use uint8 as dtype to redice the size of the dataset and speed up training.
        ```python
        def __init__(self):
            BaseFeature.__init__(self)
            self.output_shape = ()
        ```
        - override the get method and implement the data transformation from sample to the tensor. The get method can
        receive any number of parameters. If their name matches lowercased first level keys in the sample, they will be
        automatically filled with the values stored in the sample. You can also get the whole sample instance by using
        the sample parameter. Return None or BaseFeature.MISSING_VALUE in case the feature cannot be computed. If the
        output shape is (), return a scalar, otherwise return a list or numpy array of the corresponding shape.
        Note that features also can have preprocessors defined, which can be trained on all samples of the data set, and
        will be applied to the tensor or scalar returned by the get method.

        A simple example:
        ```python
        def get(self, userAccount, paymentStatistics):
            if userAccount['plan'] == 'Premium':
                return sum(x['InvoiceSum' for x in paymentStatistics])
            elif userAccount['plan'] == 'Enterprise':
                return 200000.0
            else:
                return None
        ```
        - for features of shape (2,)  or larger, you also could override getattributes to indicate the name of
        the tensor columns, whenever it makes sense. These names will then be used at several places at diagnostics
        output and in the model metrics.
    """

    MISSING_VALUE = -1

    def __init__(self):
        BaseOfBases.__init__(self)
        self.preprocessors = []
        self.dtype = np.dtype('float64')

    def getfromsample(self, sample):
        kwargs = dict()
        for param in inspect.getfullargspec(self.get).args[1:]:
            if param == 'sample':
                kwargs[param] = sample
            else:
                p = param[0].upper() + param[1:]
                if p in sample.data:
                    kwargs[param] = sample.data[p]
                else:
                    if p in [x.__name__ for x in sample.__class__.__bases__]:
                        kwargs[param] = sample
                    else:
                        if Settings.OnFeatureFailure == 'impute':
                            return BaseFeature.MISSING_VALUE
                        else:
                            raise Exception('Cannot get feature %s: no data for %s' % (self.name, p))

        return self.get(**kwargs)

    def get(self, *args, **kwargs):
        raise NotImplementedError()

    def getattributes(self):
        if self._get_width() == 1:
            return ['']
        else:
            return ['_' + str(x) for x in range(0, self._get_width())]


class TrivialFeature(BaseFeature):
    """
    Returns a numeric value stored in the sample under the defined path.
    """

    def __init__(self, path, output_shape):
        """
        :param path: key name in the sample, or a path of key names separated by dots, eg. 'User.Invoice.Total'
        :param output_shape: The shape of the feature, for example () for a scalar.
        """
        BaseFeature.__init__(self)
        self.path = path
        self.output_shape = output_shape

    def get(self, sample):
        data = sample.data
        for part in self.path.split('.'):
            if not isinstance(data, dict) or part not in data:
                return BaseFeature.MISSING_VALUE
            data = data[part]
        result = converttonumpy(data)
        if result.shape == (1,) and self.output_shape == ():  # unfold scalars to have it more comfortable when using
            # this for labels
            return result[0]
        if result.shape != self.output_shape:
            if Settings.OnFeatureFailure == 'raise':
                raise Exception('TrivialFeature shape mismatch: defined %s, but the sample has %s' % (
                    self.output_shape, result.shape))
            else:
                logging.info('TrivialFeature shape mismatch: defined %s, but the sample has %s' % (
                    self.output_shape, result.shape))
                return np.full(self.output_shape, BaseFeature.MISSING_VALUE)
        return result


class BasePreprocessor(BaseOfBases):
    """
    Preprocessors can be trained on the whole data sets and then be used to modify the tensors returned by BaseFeature
    instances, before they will be used in model (as input or labels). A typical example is normalization of the values,
    removing outliers etc. You can use all Scikit-Learn preprocessors as well as some ours, see iwlearn.preprocessors
    """

    def __init__(self):
        BaseOfBases.__init__(self)

    def calculate(self, data):
        self._calculate_impl(data)

    def _calculate_impl(self, data):
        pass

    def process(self, data):
        raise NotImplementedError()


class BasePredictionFactory(BaseOfBases):
    """
    Create an instance of Prediction based on raw scores returned by the model. Use TrivialPredictionFactory or other
    prediction factories packaged with iwlearn in the sample cases when the model output is already the value that
    you can reasonably use in further workflow.

    You might want to define your own conversion code if there is no trivial correspondence between the model score
    and the actions you'll be performing with its results. For example, if the model has to predict a conversion
    probability of users, you might want to use various incentives, banners, retargeting ads and emails depending on
    how high is the predicted probability (eg. giving 10% rebate for probabilities > 90%, send an email for 50% and
    show a retargeting ad for 30%.

    You might encode this logic somewhere in your app outside of the iwlearn, perhaps even in some other microservice.
    But then you will lose the ability to use the final model actions (GiveRebate, SendEmail, ShowRetargeting) for
    model evaluation, eg. automatically get the metrics of accuracy per action, or the number of false positives in
    the last months for the GiveRebate action.


    To implement the logic with iwlearn, you are typically to create a new Prediction class having this logic inside
    the constructor. To say the model to use this new Prediction class for predictions, you'll be
    configuring it with a new prediction factory.
    """

    def __init__(self):
        BaseOfBases.__init__(self)

    def create(self, scores, predictor):
        return BasePrediction(predictor)


class ScorePredictionFactory(BaseOfBases):
    """
    Create an instance of Prediction based on raw scores returned by the model.
    """

    def __init__(self):
        BaseOfBases.__init__(self)

    def create(self, scores, predictor):
        return ScorePrediction(scores, predictor)


class TrivialPredictionFactory(BasePredictionFactory):
    """
        Return the raw output of the model as the prediction. Because all predictions have to inherit from the
        BasePrediction or at least implement its interface, be sure that the raw model is really not a number or
        something raw. This is for example usually the case for BaseRule-based models.
    """

    def __init__(self):
        BasePredictionFactory.__init__(self)

    def create(self, scores, predictor):
        if scores is None:
            prediction = NonePrediction(predictor)
        else:
            prediction = scores
        return prediction


class BasePrediction(object):
    """
    Encapsulates the result of model prediction as well as provides a simple default conversion from the raw model
    output into a prediction class.

    You might want to define your own Prediction instances if there is no trivial correspondence between the model score
    and the actions you'll be performing with its results. For example, if the model has to predict a conversion
    probability of users, you might want to use various incentives, banners, retargeting ads and emails depending on
    how high is the predicted probability (eg. giving 10% rebate for probabilities > 90%, send an email for 50% and
    show a retargeting ad for 30%.

    You might encode this logic somewhere in your app outside of the iwlearn, perhaps even in some other microservice.
    But then you will lose the ability to use the final model actions (GiveRebate, SendEmail, ShowRetargeting) for
    model evaluation, eg. automatically get the metrics of accuracy per action, or the number of false positives in
    the last months for the GiveRebate action.

    To implement the logic with iwlearn, you are typically to create a new Prediction class having this logic inside
    of the constructor. To say the model to use this new Prediction class for predictions, you'll be
    configuring it with a new prediction factory.

    Besides the raw scores outputted by the model, the prediction can contain a list of children - the predictions
    of the sub-models that have been used to make this prediction, in case it has been made by an ensemble model.
    """

    def __init__(self, predictor=None):
        # BaseOfBases.__init__(self)
        if predictor:
            self.predictor = predictor.name
            self.task = predictor.task
            self.tag = predictor.tag
        else:
            self.predictor = None
            self.task = None
            self.tag = None

        self.children = dict()

    def set_predictor(self, predictor):
        if predictor:
            self.predictor = predictor.name
            self.task = predictor.task
            self.tag = predictor.tag

    def __str__(self):
        return '%s %s' % (self.predictor if hasattr(self, 'predictor') and self.predictor is not None else 'unkown',
                          self.prediction if hasattr(self, 'prediction') and self.prediction is not None else 'unkown')


class ScorePrediction(BasePrediction):
    def __init__(self, scores=None, predictor=None):
        BasePrediction.__init__(self, predictor)
        self.scores = scores
        if self.scores is not None:
            if np.isscalar(self.scores):
                # Typically used for regression
                self.prediction = self.scores
            else:
                if 0 < len(self.scores):
                    # Typically used for classifiers returning probabilities vector of classes
                    self.prediction = np.argmax(self.scores)


class NonePrediction(BasePrediction):
    def __init__(self, predictor):
        BasePrediction.__init__(self, predictor)


class BaseModel(BaseFeature):
    """
    Base class for models. See also BaseRule (inheriting from BaseModel) for base class for rules, as well as models
    in iwlearn.models that are prepared to use Scikit-Learn or Keras as the engines.

    You should only inherit from this class if you cannot use neither Keras nor Scikit-Learn for your model. When
    inheriting:
    - implement the constructor by setting appropriate default valies for output shape, metrics, prediction_factory, etc
    - implement train_impl, predict_impl, save_impl, and class method load
    - Also remember that models can also be used as a feature for another models, so test your design in this situation.
    """

    def __init__(self, task, features, sampletype, prediction_factory=None, labelkey=None, metrics=None, labels=None):
        """
        Models are identified by task, name and tag. In general, you might have several different approaches to solve
        the same task, and each approach evolves in time so that you need an exact tag to evaluate the performance of
        each version.

        For example, the task could be to predict the conversion probability of a user: UserConversion. The task also
        defines the default name of the label stored in the samples, as well as allows to evaluate the performance
        of the user conversion prediction system as a whole.

        One approach to solve this task could be using some rules, so you have an instance of BaseRule. Another
        approach would be  Machine Learning for the samples whenever rules are unsure about the prediction, so you
        have an instance of BaseModel. Because self.name is defaulted to be class name, these two approaches will have
        different self.name but the same self.task.

        Rules and models can be improved with time. Each saved instance of a model or rule gets a tag containing the
        hash code ot it's source code.
        :param task: The task of the model
        :param features: Features of the model
        :param sampletype: The type of the sample the model is trained for. If you want to support several sample types
        with the same model, which is quite unusual, pass None as sampletype to turn off the type checking.
        :param prediction_factory: The prediction factory to generate prediction instances
        :param labelkey: In case the label information is stored in some other key than self.task+'Label', you can pass
        the custom label key. Using the labels parameters is a more preferred way to customize labels of the model
        :param metrics: Metrics to be used during evaluation, as well as (for Keras) during the training of the model
        :param labels: Pass feature classes implementing the label extraction from the sample. You can also have
        several labels for multi-output models (Keras).
        """
        BaseFeature.__init__(self)
        self.features = features
        self.task = task
        if self.task is None or self.task == '':
            raise Exception('Unkown Task')

        self.sampletype = sampletype

        if prediction_factory is None:
            prediction_factory = BasePredictionFactory()
        self.prediction_factory = prediction_factory
        self.meta_infos = dict()
        self.metrics = metrics
        if self.metrics is None:
            self.metrics = []

        # X.shape must be equal to (number of samples, ) + model.input_shape
        # y_true.shape must be equal to (number of samples, ) + model.output_shape
        # That is, input and output shapes are per sample.

        # default suitable for classification or regression. Change in child classes if needed.
        self.output_shape = ()

        self.input_shape = BaseModel.calculate_shape(features=self.features)

        source_code = inspect.getsource(type(self))
        self.tag = hex(hash(source_code) + sys.maxsize + 1)[2:]

        if labels is not None:
            self.labels = labels
        elif labelkey == '':
            self.labels = []
        elif labelkey is None:
            self.labels = [TrivialFeature(self.task + 'Label', self.output_shape)]
        else:
            self.labels = [TrivialFeature(labelkey, self.output_shape)]

    def __repr__(self):
        return self.task + '(' + self.name + ')'

    @staticmethod
    def calculate_shape(features=None, shapes=None):
        if features is None and shapes is None:
            raise Exception('Pass either features or shapes')
        if shapes is None:
            shapes = [f.output_shape for f in features]

        input_shape = [0, ]
        for shape in shapes:
            if len(shape) == 0:
                input_shape[0] += 1
                continue
            for dim in range(0, len(shape)):
                if dim == 0:
                    input_shape[0] += shape[dim]
                else:
                    if len(input_shape) <= dim:
                        input_shape.append(shape[dim])
                    if input_shape[dim] != shape[dim]:
                        raise Exception('Shape of all features starting from the dimension 2 must match')

        if input_shape == [0]:
            return ()

        return tuple(input_shape)

    def createX(self, samples, preprocess=True, calculate_preprocessors=True):
        """
        Create X (input tensor) for the model, combining all features of the model. Usually, you don't need to call this
        """
        tensors = [create_tensor(samples, feature, preprocess, calculate_preprocessors) for feature in self.features]
        return combine_tensors(tensors, self.input_shape, len(samples))

    def create_y_true(self, samples, preprocess=True, calculate_preprocessors=True):
        """
        Create y (label tensor) for the model, combining all labels of the model. Usually, you don't need to call this.
        """
        tensors = [create_tensor(samples, label, preprocess, calculate_preprocessors) for label in self.labels]
        return combine_tensors(tensors, self.output_shape, len(samples), self.dtype)

    @staticmethod
    def getfeaturemap(features):
        """
        Get feature names for each columm of the tensor that would be produced if you combine all features together.
        Usually, you don't need to call this.
        :param features: Instances of BaseFeature
        """
        featmap = []
        for f in features:
            for a in f.getattributes():
                featmap.append(f.name + a)
        return featmap

    @classmethod
    def load(cls, filepath):
        """
        Loads a model from the filepath (string). You can pass '<your-path>/*' as filepath, in this case the latest
        model from that directory will be loaded
        """
        if filepath.endswith('/*'):
            latest_entry = None
            latest_ctime = None
            for entry in scandir(filepath[:-2]):
                if entry.name.endswith('.dill'):
                    if latest_ctime is None or entry.stat().st_ctime > latest_ctime:
                        latest_ctime = entry.stat().st_ctime
                        latest_entry = entry
            if latest_entry is None:
                raise Exception('No models found in %s' % filepath)
            filepath = latest_entry.path

        f = open(filepath, 'rb')
        result = dill.load(f)
        f.close()
        metapath = filepath.replace('.dill', '.json')
        if os.path.isfile(metapath):
            f = open(metapath, 'rb')
            result.meta_data = json.load(f)
            f.close()
        return result

    def _save(self, folder, creator):
        filepath_modelname = '%s/%s' % (folder, self.tag,)

        filepath_pickle = filepath_modelname + '.dill'
        with open(filepath_pickle, 'wb') as f:
            dill.dump(self, f)
            logging.info('Model saved to %s' % filepath_pickle)

        self.meta_infos["created"] = dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        self.meta_infos["createdby"] = creator
        self.meta_infos["task"] = self.task
        self.meta_infos["filepath"] = filepath_pickle

        filepath_json = filepath_modelname + '.json'
        with open(filepath_json, 'w') as f:
            json.dump(self.meta_infos, f, sort_keys=True, indent=4, separators=(',', ': '))
            logging.info('Meta Infos saved to %s' % filepath_json)

        return filepath_modelname

    def train(self, dataset, **configuration):
        """
        Train the model on the instance of the iwlearn.training.Dataset
        :param dataset: Training set
        :param configuration: named parameter set to pass to the engine (sklearn classifier, regressor, or Keras model)
        """
        if dataset.strsampletype != self.sampletype.__name__:
            raise Exception('Model %s cannot train on %s' % (repr(self), dataset.strsampletype))
        self.train_impl(dataset, **configuration)

    def train_impl(self, dataset, **configuration):
        """
        When overriding, implement it by training on the dataset: iwlearn.training.Dataset.

        If possible, train in batches by using the get_samples method of the dataset, to allow training on the datasets
        not fitting into the RAM. If not possible, use the get_all_samples method to retrieve the combined input and
        label tensors for the whole dataset. You should never use datasets __getitem__ method for training
        (i.e. do never ```for sample in dataset:```). This method is slow and is for debugging purposes only.
        :param dataset: training set
        :param configuration: pass this named parameters set to the underlying model (classifier, regressor, Keras
        model etc)
        """
        raise NotImplementedError()

    def predict(self, samples):
        """
        Return an instance of BasePrediction class for each passed sample in the samples list.

        This method will extract model input tensors from the data stored in the sample using self.features classes,
        pre-proccess them (eg. - normalize, remove outliers) using the preprocessors defined in per feature, pass
        the preprocessed inputs to the model, receive the raw model prediction tensor (we call it scores), and convert
        it to given Prediction class using the configured self.prediction_factory.
        :param samples: a list of one or several samples. Always pass several samples at once to speedup process with
        some model types.
        :return: An instance of BasePrediction class.
        """
        if self.sampletype is not None:
            for s in samples:
                if not isinstance(s, self.sampletype):
                    raise Exception('Model %s cannot predict on %s' % (self.tag, str(s)))
        return self.predict_impl(samples)

    def predict_impl(self, samples):
        """
        Usually you don't need to call it directly. See the predict method.
        """
        return [self.prediction_factory.create(scores, self) for scores in self.predict_scores(samples)]

    def predict_scores(self, samples):
        """
        Usually you don't need to call it directly. See the predict method.
        """
        X = self.createX(samples, calculate_preprocessors=False)
        return self._predict_scores_impl(X)

    def _predict_scores_impl(self, X):
        """
        Usually you don't need to call it directly. See the predict method.

        When implementing, return a list of raw model outputs (aka scores, y_pred) for each row in X
        """
        raise NotImplementedError()

    def getfromsample(self, sample):
        """
        This method will be used if this model is used as a feature in the greater model. When inheriting your
        model, consider also re-implementing it according to the output shape and content of the model
        """
        if self.dtype == np.object:
            return self.predict([sample])[0]
        else:
            return self.predict_scores([sample])[0]

    def get(self, *args, **kwargs):
        raise NotImplementedError('You cannot call BaseModel.get(). Did you mean to call getfromsample instead?')

    def evaluate(self, dataset):
        """
        Predict the passed test set and calculate self.metrics on the result. Depending on the particular model,
        some additional information (for example about feature importances) will be included.
        :param dataset: Train dataset, iwlearn.training.Dataset
        :return: A dictionary with metric names as keys, metric values as floats.
        """
        if self.sampletype is not None and dataset.strsampletype != self.sampletype.__name__:
            raise Exception('Model %s cannot evaluate on %s' % (self.task, dataset.strsampletype))

        if len(self.metrics) == 0:
            raise Exception('No metrics defined for the evaluation')

        X_test, y_test = dataset.get_all_samples(features=[f.name for f in self.features], label=self.labels[0].name)

        scores = self._predict_scores_impl(X_test)
        predictions = [self.prediction_factory.create(score, self) for score in scores]

        y_pred = []

        for prediction in predictions:
            if prediction is None:  # it can be None in case of Rules
                pred = BaseFeature.MISSING_VALUE
            else:
                pred = prediction.prediction if hasattr(prediction, 'prediction') else BaseFeature.MISSING_VALUE
            y_pred.append(pred)

        y_pred = np.array(y_pred)
        metricresults = {'Number of test samples': len(y_test)}
        for metricfoo in self.metrics:
            metricresults[metricfoo.__name__] = metricfoo(y_test, y_pred)

        return metricresults, y_pred, y_test

    def evaluate_and_print(self, dataset):
        """
        Predict the passed test set and calculate self.metrics on the result. Depending on the particular model,
        some additional information (for example about feature importances) will be included.
        Then print the results on the console.

        :param dataset: Train dataset, iwlearn.training.Dataset
        :return: A dictionary with metric names as keys, metric values as floats.
        """
        metricresults, y_pred, y_test = self.evaluate(dataset)

        print()
        print('Evaluation of ' + repr(self))
        print('--------------------------------------------------------------------------------')
        for temp in sorted(metricresults.keys()):
            print('%s\t%s' % (temp, str(metricresults[temp])))

        return metricresults, y_pred, y_test


class RulePrediction(BasePrediction):
    def __init__(self, prediction):
        BasePrediction.__init__(self, None)
        self.prediction = prediction


class BaseRule(BaseModel):
    """
    Rules are models with hard-coded predictions (as opposed to learned predictors). You might want to use them if
    you need to go live very quickly, or have some fixed rules (eg. laws or white/blacklists) that need to be hard-coded

    Rules should make a prediction only if they are "absolutely" or "pretty" sure about it, and return None if no
    prediction can be made.

    Inheriting your rule from this class:
    - implement the constructor to pass the list of features and the sample type, eg:
    ```python
    def __init__(self):
        BaseRule.__init__(self,
                          [
                              ff.LivingAreaMedian(),
                              ff.RoomMedian(),
                              ff.PercentageHousesForRent(),
                              ff.ExpensivePrice()],
                          'RelocationPrediction',
                          sampletype=RelocationUserSample,
                          labelkey='RelocationLabel')
    ```
    and implement the rule logic in the _implement_rule method:
    ```python
    def _implement_rule(self, livingAreaMedian, roomMedian, percentageHousesForRent, expensivePrice):
        if livingAreaMedian >= 75 and roomMedian >= 3 and \
            percentageHousesForRent > 0.3 and expensivePrice > 1.0:
            return RulePrediction(1)    # class 1 means "positive"
        return RulePrediction(0)
    ```
    The parameters of the _implement_rule method correspond to the lowercased names of the self.features without the
    "Feature" suffix.
    """

    def train_impl(self, dataset, **configuration):
        raise NotImplementedError()

    def __init__(self, features, task=None, sampletype=None, labelkey=None, metrics=None):
        """
        :param features: An array with at least one instance of BaseFeature to extract information from the sample. You
        can also use a TrivialFeature if the information from the sample don't need to be processed. On the other hand,
        it is a good practice to implement features returning numeric results, because you could then quicky go from
        a rule to a ML model reusing the same features.
        :param task: The task of the rule, similar to the tasks of models. See comment in the BaseModel class
        :param sampletype: Sample type, or None if you want to turn off the type checking
        :param labelkey: Optionally, define a custom key for the label in the sample. Even though a rule cannot be
        trained, its performance can be evaluated against the labels, and its features can be plotted per class. That's
        why we need labels here.
        :param metrics: Metrics to be used when the performance of this rule must be evaluated.
        """
        if task is None:
            task = self.__class__.__name__
        if metrics is None:
            metrics = [recall_score, precision_score, f1_score, confusion_matrix]
        BaseModel.__init__(self, task, features, sampletype, TrivialPredictionFactory(), labelkey, metrics)
        self.dtype = np.object

    def get_param(self, feature_name):
        param = feature_name[0].lower() + feature_name[1:]
        if param.endswith('Feature'):
            param = param[0:-len('Feature')]
        if param.endswith('Rule'):
            param = param[0:-len('Rule')]
        if param.endswith('Model'):
            param = param[0:-len('Model')]
        return param

    def _predict_scores_impl(self, X):
        y_pred = []
        for row in X:
            kwargs = dict()
            col = 0
            for feature in self.features:
                param = self.get_param(feature.name)
                if feature.output_shape == ():  # special handling: features returning scalars will be passed as
                    # scalars; features returning arrays will be passed as arrays
                    kwargs[param] = row[col]
                else:
                    kwargs[param] = row[col:col + feature._get_width()]
                col += feature._get_width()

            real_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, NonePrediction):
                    real_kwargs[k] = None
                else:
                    real_kwargs[k] = v
            logging.debug('%s\t%s' % (self.__class__.__name__, real_kwargs))
            prediction = self._implement_rule(**real_kwargs)
            if prediction is None:
                prediction = NonePrediction(self)
            if prediction.predictor is None:
                prediction.set_predictor(self)
            prediction.children = kwargs
            y_pred.append(prediction)

        return y_pred

    def _implement_rule(self, **kwargs):
        """
        Override in your class, define the parameters corresponding to the lowercased names of your features, and
        return either an instance of RulePrediction, or None if no prediction can be made.
        """
        raise NotImplementedError()

    def get(self, *args, **kwargs):
        raise NotImplementedError()


def converttonumpy(value):
    """
    Takes the label or feature return value, and returns np.array of it of proper shape
    """
    y = value
    if y is None:
        y = BaseFeature.MISSING_VALUE
    if isinstance(y, int) or isinstance(y, float) or isinstance(y, Decimal):
        y = [float(y)]
    return np.array(y)


class BaseDataSource(BaseOfBases):
    """
    This is a base class to define data sources. Only if you cannot inherit from any of the particular data sources
    from iwlearn.datasources, should you inherit from this class. If you're implementing access to some database not
    included in iwlearn.datasources, please consider sharing it with us by making a pull request.

    When interiting from this class, implement the constructure and optionally the makeimpl method.
    """

    def __init__(self):
        BaseOfBases.__init__(self)

    def make(self, **kwargs):
        started = dt.datetime.now()
        result = None
        try:
            result = self.makeimpl(**kwargs)
            if result is not None:
                logging.info(
                    '%s retrieved in %.0f ms' % (self.name, (dt.datetime.now() - started).total_seconds() * 1000))
            else:
                logging.info('Cannot retrieve  %s ' % self.name)
        except:
            logging.error('Cannot retrieve sample %s ' % self)
        return result

    def makeimpl(self, **kwargs):
        """
        Override and return a dictionary with the raw data retrieved from the data source, or None if there is no
        possibility to retrieve.
        """
        raise NotImplementedError()

    def getfromsample(self, sample):
        key = self.name
        if key.endswith('DataSource'):
            key = key[:-len('DataSource')]
        if key.endswith('Sample'):
            key = key[:-len('Sample')]
        if key not in sample:
            return BaseFeature.MISSING_VALUE
        return sample[key]


class Settings(object):
    """If a feature cannot be computed from a sample, whether to raise an Exception, or to replace it with
    BaseFeature.MISSING_VALUE"""
    OnFeatureFailure = 'raise'  # or 'impute'

    """If a feature is not present in the Dataset, whether to raise an Exception, or to replace it with 
    BaseFeature.MISSING_VALUE"""
    OnFeatureMissing = 'impute'  # or 'raise'

    """Whether to raise exception if the dataset contains samples with not-unique entityid"""
    EnsureUniqueEntitiesInDataset = False

    """If a feature gets computed, whether to log the execution time or not."""
    LogFeatureExecutionTime = False

    """Cannot get feature, whether to log the exeception or not."""
    LogCannotGetFeatureError = True


def create_tensor(samples, feature, preprocess=True, calculate_preprocessors=True, dtype=None):
    resultingshape = (len(samples),) + feature.output_shape
    logging.info('Creating tensor of shape %s' % (str(resultingshape)))
    if hasattr(feature, 'dtype'):
        if dtype is None or dtype == feature.dtype:
            dtype = feature.dtype
        else:
            raise Exception('Features define contradicting dtype')
    if dtype is None:
        dtype = np.float32

    X = np.full(resultingshape, BaseFeature.MISSING_VALUE, dtype=dtype)
    row = 0

    if Settings.LogFeatureExecutionTime:
        tic = time.time()

    for sample in samples:
        try:
            val = feature.getfromsample(sample)
            if dtype != np.object:
                val = converttonumpy(val)
            X[row] = val
        except:
            if Settings.LogCannotGetFeatureError:
                logging.exception('Cannot get feature %s' % (feature.name,))
        row += 1

    if Settings.LogFeatureExecutionTime:
        toc = time.time()
        logging.info(
            '%s took %s total, ~%s ms/sample' % (feature.name, 1000 * (toc - tic), (1000 * (toc - tic)) / len(samples)))

    logging.debug(X)

    if preprocess and hasattr(feature, 'preprocessors'):
        for preprocessor in feature.preprocessors:
            if calculate_preprocessors:
                preprocessor.calculate(X)
            preprocessor.process(X)

    return X


def combine_tensors(tensors, sampleshape, numsamples, dtype=None):
    resultingshape = (numsamples,) + sampleshape

    # shortcut for just one feature - useful when working with images
    if len(tensors) == 1 and tensors[0].shape == resultingshape:
        return tensors[0]

    logging.debug('Combining tensors to shape %s' % (str(resultingshape)))

    for tensor in tensors:
        if dtype is None or dtype == tensor.dtype:
            if hasattr(tensor[0], 'dtype'):
                dtype = tensor[0].dtype
            else:
                dtype = np.object
        else:
            raise Exception('Tensors define contradicting dtype')
    if dtype is None:
        dtype = np.float32

    X = np.full(resultingshape, BaseFeature.MISSING_VALUE, dtype=dtype)

    if len(tensors) == 0 or len(tensors[0]) == 0:
        return X

    if len(resultingshape) > 1:
        for row in range(0, len(tensors[0])):
            _broadcastvalue(X[row], [x[row] for x in tensors], dtype, resultingshape)
    else:
        for tensor in tensors:
            _broadcastvalue(X, tensor, dtype, resultingshape)

    logging.debug(X)

    return X


def _broadcastvalue(target, values, dtype, resultingshape):
    col = 0
    for val in values:
        if np.isscalar(val) or dtype == np.object:
            try:
                target[col] = val
            except:
                pass
            col += 1
        else:
            try:
                broadcastshape = list(resultingshape[1:])
                broadcastshape[0] = len(val)
                val = _agreeshape(val, broadcastshape)
                target[col:col + len(val)] = val
            except:
                pass

            col += len(val)


def _agreeshape(val, broadcastshape):
    try:
        val = np.array(val)
        if val.shape != broadcastshape:
            if len(val.shape) == 1 and val.shape[0] > 1:
                val = np.broadcast_to(val, reversed(broadcastshape))
                val = np.reshape(val, broadcastshape)
            if len(val.shape) != 1:
                val = np.broadcast_to(val, broadcastshape)
        return val
    except:
        pass
