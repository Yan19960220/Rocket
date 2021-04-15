# -*- coding: utf-8 -*-
# @Time    : 4/14/21 7:43 PM
# @Author  : Yan
# @Site    : 
# @File    : utils.py
# @Software: PyCharm

import argparse
import numpy as np


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-u", "--update_data", type=bool, default=False)
    parser.add_argument("-n", "--num_runs", type=int, default=10)
    parser.add_argument("-k", "--num_kernels", type=int, default=10_000)
    return parser.parse_args()


def file2list(path):
    with open(path, "r") as f:
        lines = [int(line.strip()) for line in f]
    return lines


def list2file(path, list_a):
    with open(path, "w") as f:
        for item in list_a:
            f.write("%s\n" % str(item))


def normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma