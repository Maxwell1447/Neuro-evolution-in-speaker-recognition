"""
Has the built-in activation functions,
code for using them,
and code for adding new user-defined ones
"""
from __future__ import division
import torch
import types


def sigmoid_activation(z):
    z = torch.clamp(z, -60, 60)
    return torch.sigmoid(z)


def tanh_activation(z):
    z = torch.clamp(2.5 * z, -60, 60)
    return torch.tanh(z)


def sin_activation(z):
    z = torch.clamp(5 * z, -60, 60)
    return torch.sin(z)


def gauss_activation(z):
    z = torch.clamp(z, -3.4, 3.4)
    return torch.exp(-5.0 * z**2)


def relu_activation(z):
    return z if z > 0.0 else torch.tensor(0.)


def softplus_activation(z):
    z = torch.clamp(5 * z, -60, 60)
    return 0.2 * torch.log(1 + torch.exp(z))


def identity_activation(z):
    return z


def clamped_activation(z):
    return torch.clamp(z, -1, 1)


def inv_activation(z):
    try:
        z = 1.0 / z
    except ArithmeticError:  # handle overflows
        return 0.0
    else:
        return z


def log_activation(z):
    z = torch.clamp(z, min=1e-7)
    return torch.log(z)


def exp_activation(z):
    z = torch.clamp(z, -60, 60)
    return torch.exp(z)


def abs_activation(z):
    return torch.abs(z)


def hat_activation(z):
    return torch.clamp(torch.tensor(1.) - torch.abs(z), min=0)


def square_activation(z):
    return z ** 2


def cube_activation(z):
    return z ** 3


class InvalidActivationFunction(TypeError):
    pass


def validate_activation(function):
    ...


class ActivationFunctionSet(object):
    """
    Contains the list of current valid activation functions,
    including methods for adding and getting them.
    """
    def __init__(self):
        self.functions = {}
        self.add('sigmoid', sigmoid_activation)
        self.add('tanh', tanh_activation)
        self.add('sin', sin_activation)
        self.add('gauss', gauss_activation)
        self.add('relu', relu_activation)
        self.add('softplus', softplus_activation)
        self.add('identity', identity_activation)
        self.add('clamped', clamped_activation)
        self.add('inv', inv_activation)
        self.add('log', log_activation)
        self.add('exp', exp_activation)
        self.add('abs', abs_activation)
        self.add('hat', hat_activation)
        self.add('square', square_activation)
        self.add('cube', cube_activation)

    def add(self, name, function):
        validate_activation(function)
        self.functions[name] = function

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise InvalidActivationFunction("No such activation function: {0!r}".format(name))

        return f

    def is_valid(self, name):
        return name in self.functions
