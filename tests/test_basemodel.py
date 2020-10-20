# -*- coding: utf-8 -*-
import pytest
import numpy as np

from iwlearn.base import _agreeshape


class TestBaseModel(object):
    def testshapeagreement(self):
        val = [1, 2, 3]
        broadcastshape = [3]
        val = _agreeshape(val, broadcastshape)
        assert val.shape == (3,)

        val = [1, 2, 3]
        broadcastshape = [3, 10]
        val = _agreeshape(val, broadcastshape)
        assert val.shape == (3, 10)

        val = [1, 2, 3]
        broadcastshape = [3, 200, 200, 2]
        val = _agreeshape(val, broadcastshape)
        assert val.shape == (3, 200, 200, 2)


if __name__ == "__main__":
    pytest.main([__file__])
