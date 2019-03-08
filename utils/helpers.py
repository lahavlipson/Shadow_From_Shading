# -*- coding: utf-8 -*-

import argparse
import random

###
# Experiment configuration parser.
###

def define_parser():
    parser = argparse.ArgumentParser()
    # training related arguments
    parser.add_argument(
        '--ep_len', help='length of each epoch', default=100, type=int)
    parser.add_argument(
        '--niter', help='# epochs to train', default=100, type=int)
    parser.add_argument(
        '--batch_size', help='input batch size', default=16, type=int)
    parser.add_argument(
        '--res_dir', help='location to write results', default="results", type=str)
    parser.add_argument(
        '--workers', help='# data loader workers', default=4, type=int)
    parser.add_argument(
        '--lr', help='learning rate for the optimizer', default=0.001, type=float)
    parser.add_argument(
        '--cuda', action='store_true', help='whether to use GPU')
    return parser


def mean(l):
    return sum(l)/len(l)