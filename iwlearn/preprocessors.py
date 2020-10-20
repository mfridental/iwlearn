import warnings
import numpy as np
from base import BasePreprocessor


class ScikitLearnPreprocessor(BasePreprocessor):
    def __init__(self, preprocessor):
        BasePreprocessor.__init__(self)

        process_fn = getattr(preprocessor, 'fit', None)
        if process_fn is None or not callable(process_fn):
            raise Exception('preprocessor must have a fit() function')

        copy = getattr(preprocessor, 'copy', None)
        if copy is None:
            warnings.warn("preprocessor '%s' doesn't have a copy attribute, it might create a copy of data."
                          "BasePreprocessors must always modify the passed data object, not copy it."
                          % type(preprocessor).__name__)

        preprocessor.copy = False
        self._preprocessor = preprocessor

    def calculate(self, data):
        fit_fn = getattr(self._preprocessor, 'fit', None)
        if fit_fn is not None and callable(fit_fn):
            fit_fn(data)
        else:
            warnings.warn("preprocessor '%s' doesn't have a fit function, won't train preprocessor"
                          % type(self._preprocessor).__name__)

    def process(self, data):
        before = id(data)
        result = self._preprocessor.transform(data)
        after = id(result)

        if result is not None and before != after:
            raise Exception("preprocessor '%s' created a copy of 'data'. It must modify the passed object instead."
                            % type(self._preprocessor).__name__)


class FilterExtremesPreprocessor(BasePreprocessor):
    def __init__(self, missing_value, lower_q=None, upper_q=None, min_val=None, max_val=None, mode='outer',
                 equal='include'):
        """
        Preprocesses a feature by replacing 'extreme' values by a default value.
        :param missing_value: Value to replace 'invalid' values with
        :param lower_q: Lower quantile below which values should be replaced by replace_value
        :param upper_q: Upper quantile above which values should be replaced by replace_value
        :param min_val: Absolute minimum allowed value
        :param max_val: Absolute maximum allowed value
        :param mode: 'outer' or 'inner' - If both the quantile and an absolute value are defined, the preprocessor will
        choose the more 'liberal' value in the case of 'outer' (i.e. min(lower_q, min_val) or max(upper_q, max_val) or
        the more 'restrictive' value in the case of 'inner' (i.e. max(lower_q, min_val) or min(upper_q, max_val)
        :param equal: 'include' or 'exclude' - if the feature's value is equal to the high or low value, either include
        it or exclude (replace it with missing_value) it
        """
        BasePreprocessor.__init__(self)

        if not (lower_q or upper_q or min_val or max_val):
            raise Exception('At least one extreme value must be specified')

        if mode not in ['outer', 'inner'] and (lower_q and min_val) or (upper_q and max_val):
            raise Exception("mode must either be 'outer' or 'inner'")

        if equal not in ['include', 'exclude']:
            raise Exception("equal must either be 'include' or 'exclude'")

        self._missing_value = missing_value
        self._lower_q = lower_q
        self._upper_q = upper_q
        self._min_val = min_val
        self._max_val = max_val
        self._mode = mode
        self._equal = equal

        self._low = None
        self._high = None

    def _get_val(self, data, q, m, t):
        v = None

        if q:
            v = np.quantile(data[data != self._missing_value], q)

        if m:
            if v is not None:
                if t == 'min':
                    return np.minimum(m, v) if self._mode == 'outer' else np.maximum(m, v)
                elif t == 'max':
                    return np.maximum(m, v) if self._mode == 'outer' else np.minimum(m, v)

            return m

        return v

    def _calculate_impl(self, data):
        if len(np.squeeze(data).shape) > 1:
            raise Exception('Only one-dimensional data is supported by FilterExtremesPreprocessor')

        self._low = self._get_val(data, self._lower_q, self._min_val, 'min')
        self._high = self._get_val(data, self._upper_q, self._max_val, 'max')

    def process(self, data):
        # TODO: Instead of replacing with missing_value, completely omit the sample
        # TODO: save and check shape
        if self._low:
            if self._equal == 'include':
                data[data < self._low] = self._missing_value
            elif self._equal == 'exclude':
                data[data <= self._low] = self._missing_value

        if self._high:
            if self._equal == 'include':
                data[data > self._high] = self._missing_value
            elif self._equal == 'exclude':
                data[data >= self._high] = self._missing_value


class MinMaxPreprocessor(BasePreprocessor):
    def __init__(self, exclude=None, equal=0):
        BasePreprocessor.__init__(self)

        self._exclude = exclude
        self._equal = equal

        self._min = None
        self._max = None

    def _calculate_impl(self, data):
        subset = data[data != self._exclude] if self._exclude is not None else data
        self._min = np.min(subset)
        self._max = np.max(subset)

    def process(self, data):
        if self._max - self._min == 0:
            if self._exclude is not None:
                data[data != self._exclude] = self._equal
            else:
                data.fill(self._equal)
        else:
            if self._exclude is not None:
                data[data != self._exclude] = (1. * data[data != self._exclude] - self._min) / (self._max - self._min)
            else:
                np.divide(1. * data - self._min, self._max - self._min, out=data)


class MeanStdPreprocessor(BasePreprocessor):
    def __init__(self, exclude, equal=0):
        BasePreprocessor.__init__(self)
        self._exclude = exclude
        self._equal = equal

        self._mean = None
        self._std = None

    def _calculate_impl(self, data):
        subset = data[data != self._exclude] if self._exclude is not None else data
        self._mean = np.mean(subset)
        self._std = np.std(subset)

    def process(self, data):
        if self._std == 0:
            data.fill(self._equal)
        else:
            np.subtract(data, self._mean, out=data)
            np.divide(data, self._std, out=data)
