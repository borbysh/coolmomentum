""" This is a modified version of 
    https://github.com/ifeherva/optimizer-benchmark/blob/master/optimizers/__init__.py   """

import argparse
import torch.optim as optim
import math


from .coolmom_pytorch import Coolmomentum

__all__ = ['parse_optimizer', 'supported_optimizers']

optimizer_defaults = {
    'coolmomentum': (Coolmomentum, 'Coolmomentum', {
        'lr': 0.01,
        'momentum': 0.99,
        'weight_decay': 5e-4,
        'beta': (1 - 0.99)**(1/(200*math.ceil(50000/128))),
        'dropout': 0.0,
    })
}


def supported_optimizers():
    return list(optimizer_defaults.keys())


def required_length(nargs):
    class RequiredLength(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if len(values) != nargs:
                msg = 'argument "{}" requires exactly {} arguments'.format(self.dest, nargs)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)

    return RequiredLength


def parse_optim_args(args, default_args):
    parser = argparse.ArgumentParser(description='Optimizer parser')
    for k, v in default_args.items():
        if type(v) == bool:
            kwargs = {'action': 'store_false' if v else 'store_true'}
        elif type(v) == list:
            kwargs = {'type': type(v[0]), 'nargs': '+', 'default': v}
        elif type(v) == tuple:
            kwargs = {'type': type(v[0]), 'nargs': '+', 'action': required_length(len(v)), 'default': v}
        else:
            kwargs = {'type': type(v), 'default': v}
        parser.add_argument('--{}'.format(k), **kwargs)
    opt = parser.parse_args(args)

    opt_params_name = ''
    for k, v in default_args.items():
        if opt.__getattribute__(k) != v:
            param_format = '' if type(v) == bool else '_{}'.format(opt.__getattribute__(k))
            opt_params_name += '_{}{}'.format(k, param_format)

    return opt, opt_params_name


def parse_optimizer(optimizer, optim_args, model_params):
    if optimizer not in optimizer_defaults:
        raise RuntimeError('Optimizer {} is not supported'.format(optimizer))

    optim_func, optim_name, def_params = optimizer_defaults[optimizer]

    optim_opts, opt_name = parse_optim_args(optim_args, def_params)
    opt_name = '{}{}'.format(optim_name, opt_name)
    return optim_func(model_params, **vars(optim_opts)), opt_name
