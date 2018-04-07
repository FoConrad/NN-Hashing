#!/usr/bin/env python
import argparse
from functools import partial

import numpy
import torch

from models import RNN
from attacks import L2Attack

def get_args():
    parser = argparse.ArgumentParser(description='Attack RNN hasher.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('target', type=str,
            help='target file for target hash value')
    parser.add_argument('--source', type=str, default=None,
            help='source file to be transformed to have target hash')
    parser.add_argument('--output', type=str, default=None,
            help='source file to be transformed to have target hash')

    parser.add_argument('--adam-lr', type=float, default=1.,
            help='learning rate for adam optimizer')
    parser.add_argument('--reg-coeff', type=float, default=0.0045,
            help='coefficient for loss regularization')
    parser.add_argument('--iters', type=int, default=0,
            help='number of steps (default is infinite loop)')

    parser.add_argument('--cuda', action='store_true',
            help='use GPU or not')
    parser.add_argument('--no-seed', action='store_true',
            help='skipping seeding random number generators')

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    if not args.no_seed:
        torch.manual_seed(42)
        numpy.random.seed(42)
        if args.cuda:
            torch.cuda.manual_seed(42)

    _cudize = lambda in_: in_.cuda() if args.cuda else in_
    att = L2Attack(model_class=partial(RNN, 512, [1204, 2048], 128, _cudize), 
                   target=args.target,
                   source=args.source, 
                   output=args.output, 
                   adam_lr=args.adam_lr,
                   reg_coeff=args.reg_coeff)
    if args.source:
        att.iterate(args.iters)
    else:
        att.query()

if __name__ == '__main__':
    main()
