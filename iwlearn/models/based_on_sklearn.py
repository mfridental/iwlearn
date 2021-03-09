# -*- coding: utf-8 -*-
import logging
import sys
from functools import cmp_to_key

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, anderson_ksamp
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import recall_score, precision_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import learning_curve, validation_curve, ShuffleSplit, ParameterGrid, ParameterSampler, \
    StratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from iwlearn.base import BaseModel, ScorePredictionFactory


def stratfiedTrainTestSplit(X, y, test_size=0.25, verbose=False):
    c0_indices = np.where(y == 1)
    c1_indices = np.where(y == 0)

    class1 = train_test_split(X[c1_indices], y[c1_indices], test_size=test_size)
    class0 = train_test_split(X[c0_indices], y[c0_indices], test_size=test_size)

    result = []
    for c0, c1 in zip(class0, class1):
        result.append(np.concatenate([c0, c1]))

    XTrain, yTrain = shuffle(result[0], result[2])
    XTest, yTest = shuffle(result[1], result[3])

    if verbose:
        logging.info('StratfiedTrainTestSplit')
        logging.info('#X: %d' % X.shape[0])
        logging.info('#y: %d' % y.shape[0])
        logging.info('#class0: %d' % c0_indices[0].shape[0])
        logging.info('#class1: %d' % c1_indices[0].shape[0])
        logging.info('#XTrain: %d' % XTrain.shape[0])
        logging.info('#yTrain: %d' % yTrain.shape[0])
        logging.info('#XTest: %d' % XTest.shape[0])
        logging.info('#yTest: %d' % yTest.shape[0])

    return XTrain, XTest, yTrain, yTest


class ScikitLearnModel(BaseModel):
    def __init__(self, task, features, sampletype, prediction_factory=ScorePredictionFactory(),
                 classifier=RandomForestClassifier(n_estimators=100, n_jobs=-1),
                 scaler=MinMaxScaler(), labelkey=None, metrics=None, labels=None):
        BaseModel.__init__(self, task, features, sampletype, prediction_factory, labelkey, metrics, labels)

        self.classifier = classifier
        self.scaler = scaler
        self.goodnessfitinfo = [{}] * self.input_shape[0]
        if metrics is None and isinstance(self.classifier, RandomForestClassifier):
            self.metrics = [recall_score, precision_score, f1_score]
        else:
            self.metrics = metrics

    def train_impl(self, dataset, **configuration):
        X_train, y_train = dataset.get_all_samples(features=[f.name for f in self.features], label=self.labels[0].name)

        setsize = len(y_train)
        if 1000 < setsize:
            setsize = 1000

        try:
            for i in range(0, X_train.shape[1]):
                self.goodnessfitinfo[i] = {
                    'min': np.min(X_train[:, i]),
                    'max': np.max(X_train[:, i]),
                    'var': np.var(X_train[:, i]),
                    'x_train': X_train[-setsize - 1:-1, i],
                    'y_train': y_train[-setsize - 1:-1]}
        except:
            logging.error('cannot calculate goodnessoffitinfo')
        return self.retrain(X_train, y_train, **configuration)

    def checkgoodnessoffit(self, samples):
        """
        This method solves the following problem: the model is created with historical training set, then
        used for inference in production. With time, the distribution of features in production can change,
        (for example because user behavior changes), so that the model is not optimal any more, because
        it has to inference on data having distributions that it didn't see during the training. We want to
        detect this situation, by storing distributions of each feature during the training in the model.
        This method calculates the distributions for the features based on passed samples, and compares them
        with the original distributions from the training time using the Cramer von Mises test.

        Another use case for the method is detecting missing data or bugs in data sources (eg. one established
        data source suddenly stop to deliver data or delivering different data)
        """

        results = {}
        X = self.createX(samples)
        y_true = self.create_y_true(samples)

        for label in set(y_true):
            XX = X[y_true == label]

            labelresult = {}

            for i in range(0, len(self.features)):
                info = self.goodnessfitinfo[i]
                x_train = info['x_train']
                y_train = info['y_train']
                x_train = x_train[y_train == label]
                x_pred = XX[:, i]

                _, pvalue = ks_2samp(x_pred, x_train)
                labelresult[self.features[i].name + ' Kolmogorov-Smirnov p'] = pvalue
                test, __, _ = anderson_ksamp([x_train, x_pred], False)
                labelresult[self.features[
                                i].name + ' Anderson-Darling stat'] = test  # 0.3 is 25% chance of having same
                # distribution, 3 is 1% chance of having same distribution
                labelresult[self.features[i].name + ' minmin'] = (info['min'], np.min(x_pred))
                labelresult[self.features[i].name + ' maxmax'] = (info['max'], np.max(x_pred))

            results['Class %d' % label] = labelresult

        return results

    def retrain(self, X_train, y_train, **configuration):
        logging.info('Numpyset X %s, y_true %s' % (X_train.shape, y_train.shape))
        logging.info(
            'Unique values per column: %s' % ([len(set(X_train[:, col])) for col in range(0, X_train.shape[1])]))
        logging.info('Training start')
        if self.scaler is None:
            X_train_scaled = X_train
        else:
            X_train_scaled = self.scaler.fit_transform(X_train)
        self.classifier.fit(X_train_scaled, np.ravel(y_train))
        return X_train, y_train

    def _predict_scores_impl(self, X):
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.classifier.predict_proba(X)

    def evaluate(self, dataset):
        metrics, y_pred, y_test = BaseModel.evaluate(self, dataset)
        featurerepl = []
        for f in self.features:
            featurerepl.extend(['%s%s' % (f.name, f.getattributes()[i]) for i in range(0, f._get_width())])
        if hasattr(self.classifier, 'feature_importances_'):
            for imp, f in zip(self.classifier.feature_importances_, featurerepl):
                metrics['%s importance' % (f)] = imp
        if hasattr(self.classifier, 'n_estimators'):
            metrics['n_estimators'] = self.classifier.n_estimators
        if hasattr(self.classifier, 'max_features'):
            metrics['max_features'] = self.classifier.max_features
        if hasattr(self.classifier, 'min_samples_leaf'):
            metrics['min_samples_leaf'] = self.classifier.min_samples_leaf
        self.meta_infos['EvaluationResult'] = metrics
        return metrics, y_pred, y_test

    def getPrediction(self, predictedproba, lamda=0.5):
        p = np.delete(predictedproba, np.s_[:-1], 1)
        p[p >= lamda] = 1
        p[p < lamda] = 0
        return p.flatten()

    def train_with_hyperparameters_optimization(self, dataset, n_splits, param_distributions,
                                                params_distributions_test_proportion, test_size):
        X, y = dataset.get_all_samples(features=[f.name for f in self.features], label=self.labels[0].name)

        X_train, X_eval, y_train, y_eval = stratfiedTrainTestSplit(X, y, test_size)
        logging.info('Numpyset X %s, y_true %s' % (X_train.shape, y_train.shape))
        logging.info(
            'Unique values per column: %s' % ([len(set(X_train[:, col])) for col in range(0, X_train.shape[1])]))
        logging.info('Training start')

        best_params = []
        outer_best_params = []

        n_iter = int(len(list(ParameterGrid(param_distributions))) * params_distributions_test_proportion)

        # just a sample
        candidates = list(ParameterSampler(param_distributions, n_iter))
        for candidate in candidates:
            logging.info(candidate)

        clf = self.classifier
        scaler = self.scaler

        logging.info("Anzahl Features: %d" % (X_train.shape[1]))
        logging.info("Anzahl Optimierungsbeispiel: %d" % (X_train.shape[0]))
        logging.info("Anzahl Evaluierungsbeispiele: %d" % (X_eval.shape[0]))

        outer_skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

        logging.info('outer scores')
        for outer_train_index, outer_test_index in outer_skf.split(X_train, y_train):
            X_outer_train = X_train[outer_train_index]
            y_outer_train = y_train[outer_train_index]
            X_outer_test = X_train[outer_test_index]
            y_outer_test = y_train[outer_test_index]

            inner_skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
            rscv = RandomizedSearchCV(estimator=clf,
                                      param_distributions=param_distributions,
                                      n_iter=n_iter,
                                      n_jobs=-1,
                                      cv=inner_skf,
                                      scoring='f1')
            scaler.fit_transform(X_outer_train)
            scaler.transform(X_outer_test)
            rscv.fit(X_outer_train, y_outer_train)

            # logging.info(rscv.cv_results_)

            test_clf = clone(clf).set_params(**rscv.best_params_)
            test_clf.fit(X_outer_train, y_outer_train)
            y_outer_predict = test_clf.predict_proba(X_outer_test)
            scores = precision_recall_fscore_support(y_outer_test, self.getPrediction(y_outer_predict))

            logging.info(rscv.best_params_)
            logging.info(scores)
            outer_best_params.append(rscv.best_params_)

        best_score = 0.01
        best_scores = []
        best_cls = self.classifier
        logging.info('eval scores')
        for best_params in outer_best_params:
            eval_clf = clone(clf).set_params(**best_params)
            scaler.fit(X_train)
            X_train_2 = scaler.transform(X_train)
            X_eval_2 = scaler.transform(X_eval)
            eval_clf.fit(X_train_2, y_train)
            y_eval_predict = eval_clf.predict_proba(X_eval_2)
            scores = precision_recall_fscore_support(y_eval, self.getPrediction(y_eval_predict))

            logging.info(best_params)
            logging.info(scores)

            if best_score < scores[2][0]:
                best_score = scores[2][0]
                best_scores = scores
                best_cls = eval_clf

        logging.info("winner")
        logging.info(best_params)
        logging.info(best_scores)
        self.scaler = scaler
        self.classifier = best_cls

    def feature_selection(self, dataset, step, n_splits):
        X_train, y_train = dataset.get_all_samples(features=[f.name for f in self.features], label=self.labels[0].name)
        rfecv = RFECV(estimator=self.classifier, step=step, cv=StratifiedKFold(n_splits))
        rfecv.fit(X_train, y_train)
        logging.info("Recursive Feature Elimination CV")
        logging.info("Optimale Anzahl an Features: %d" % rfecv.n_features_)
        logging.info("Optimale Feature: %s" % str(rfecv.ranking_))

        return rfecv.ranking_

    def plot_learning_curve(self, dataset, test_size=0.25, runs=10, learningstep=100, scoring=None):
        X, y = dataset.get_all_samples(features=[f.name for f in self.features], label=self.labels[0].name)
        plt.figure()
        plt.title('learning curve')
        plt.xlabel('Training Samples')
        plt.ylabel('Score')
        cv = ShuffleSplit(n_splits=runs, test_size=test_size, random_state=0)
        train_sizes = np.linspace(0.01, 1.0, learningstep)
        train_sizes, train_scores, test_scores = learning_curve(estimator=self.classifier, X=X, y=y,
                                                                train_sizes=train_sizes, cv=cv, n_jobs=1,
                                                                scoring=scoring)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1,
                         color='r')
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                         color='g')
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Scores")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Test Scores")
        plt.legend(loc="best")
        return plt

    def plot_validation_curve(self, dataset, param_name=None, param_range=None, test_size=0.25, runs=10):
        X, y = dataset.get_all_samples(features=[f.name for f in self.features], label=self.labels[0].name)
        plt.figure()
        plt.title('Validation Curve - %s' % (param_name,))
        plt.xlabel(param_name)
        plt.ylabel('Score')
        cv = ShuffleSplit(n_splits=runs, test_size=test_size, random_state=0)
        train_scores, test_scores = validation_curve(estimator=self.classifier, X=X, y=y, param_name=param_name,
                                                     param_range=param_range, cv=cv, n_jobs=-1)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1,
                         color='r')
        plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                         color='g')
        plt.plot(param_range, train_scores_mean, 'o-', color="r", label="Training Scores")
        plt.plot(param_range, test_scores_mean, 'o-', color="g", label="Test Scores")
        plt.legend(loc="best")
        return plt


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
