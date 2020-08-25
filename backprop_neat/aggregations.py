"""
Has the built-in aggregation functions, code for using them,
and code for adding new user-defined ones.
"""

import types
import warnings
import torch


def product_aggregation(x):  # note: `x` is a list or other iterable
    return torch.prod(x, dim=0)


def sum_aggregation(x):
    return torch.sum(x, dim=0)


def max_aggregation(x):
    return torch.max(x, dim=0)


def min_aggregation(x):
    return torch.min(x, dim=0)


def maxabs_aggregation(x):
    return torch.max(torch.abs(x), dim=0) * torch.sign(x)


def median_aggregation(x):
    return torch.median(x, dim=0)


def mean_aggregation(x):
    return torch.mean(x, dim=0)


class InvalidAggregationFunction(TypeError):
    pass


def validate_aggregation(function):
    ...


class AggregationFunctionSet(object):
    """Contains aggregation functions and methods to add and retrieve them."""

    def __init__(self):
        self.functions = {}
        self.add('product', product_aggregation)
        self.add('sum', sum_aggregation)
        self.add('max', max_aggregation)
        self.add('min', min_aggregation)
        self.add('maxabs', maxabs_aggregation)
        self.add('median', median_aggregation)
        self.add('mean', mean_aggregation)

    def add(self, name, function):
        validate_aggregation(function)
        self.functions[name] = function

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise InvalidAggregationFunction("No such aggregation function: {0!r}".format(name))

        return f

    def __getitem__(self, index):
        warnings.warn("Use get, not indexing ([{!r}]), for aggregation functions".format(index),
                      DeprecationWarning)
        return self.get(index)

    def is_valid(self, name):
        return name in self.functions
