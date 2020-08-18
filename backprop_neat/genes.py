"""Handles node and connection genes."""
import warnings
from random import random

import torch
from backprop_neat.attributes import FloatAttribute, BoolAttribute, StringAttribute


def tensorize(param):
    if isinstance(param, torch.Tensor):
        return param.clone().detach().requires_grad_(True)
    return param


def tensor_copy(param):
    if isinstance(param, torch.Tensor):
        return param.clone().detach().requires_grad_(True)
    return param


class BaseGene(object):
    """
    Handles functions shared by multiple types of genes (both node and connection),
    including crossover and calling mutation methods.
    """
    def __init__(self, key):
        self.key = key

    def __str__(self):
        attrib = ['key'] + [a.name for a in self._gene_attributes]
        attrib = ['{0}={1}'.format(a, getattr(self, a)) for a in attrib]
        return '{0}({1})'.format(self.__class__.__name__, ", ".join(attrib))

    def __lt__(self, other):
        assert isinstance(self.key,type(other.key)), "Cannot compare keys {0!r} and {1!r}".format(self.key,other.key)
        return self.key < other.key

    @classmethod
    def parse_config(cls, config, param_dict):
        pass

    @classmethod
    def get_config_params(cls):
        params = []
        if not hasattr(cls, '_gene_attributes'):
            setattr(cls, '_gene_attributes', getattr(cls, '__gene_attributes__'))
            warnings.warn(
                "Class '{!s}' {!r} needs '_gene_attributes' not '__gene_attributes__'".format(
                    cls.__name__,cls),
                DeprecationWarning)
        for a in cls._gene_attributes:
            params += a.get_config_params()
        return params

    def init_attributes(self, config):
        for a in self._gene_attributes:
            setattr(self, a.name, a.init_value(config))

    def mutate(self, config):
        for a in self._gene_attributes:
            v = getattr(self, a.name)
            setattr(self, a.name, a.mutate_value(v, config))

    def clamp(self, config):
        for a in self._gene_attributes:
            v = getattr(self, a.name)
            if isinstance(v, torch.Tensor):
                v = v.detach().item()
                setattr(self, a.name, torch.tensor(a.clamp(v, config), requires_grad=True))

    def copy(self):
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            setattr(new_gene, a.name, tensor_copy(getattr(self, a.name)))

        return new_gene

    def crossover(self, gene2):
        """ Creates a new gene randomly inheriting attributes from its parents."""
        assert self.key == gene2.key

        # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
        # here because `choice` is substantially slower.
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            if random() > 0.5:
                setattr(new_gene, a.name, tensorize(getattr(self, a.name)))
            else:
                setattr(new_gene, a.name, tensorize(getattr(gene2, a.name)))

        return new_gene


class DefaultNodeGene(BaseGene):
    _gene_attributes = [FloatAttribute('bias'),
                        StringAttribute('activation', options='sigmoid'),
                        StringAttribute('aggregation', options='sum')]

    def __init__(self, key):
        assert isinstance(key, int), "DefaultNodeGene key must be an int, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = torch.abs(self.bias.detach() - other.bias.detach()).item()
        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        return d * config.compatibility_weight_coefficient


class DefaultConnectionGene(BaseGene):
    _gene_attributes = [FloatAttribute('weight'),
                        BoolAttribute('enabled')]

    def __init__(self, key):
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = torch.abs(self.weight - other.weight)
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient

