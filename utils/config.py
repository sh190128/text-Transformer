# -*- coding: utf-8 -*-
# @Time    : 2023/2/13 20:49
# @Author  : xsy
# @Project ：TST-mimic 
# @File    : config.py

import argparse


def config():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_path', default='./data')
    parser.add_argument('-load_model_weights', default=None)

    parser.add_argument('-epoch', type=int, default=200)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-lr', type=float, default=1e-5)
    parser.add_argument('-val_ratio', type=float, default=0.1)

    parser.add_argument('-max_len', type=int, default=2048)
    parser.add_argument('-feat_dim', type=int, default=18)
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-n_heads', type=int, default=8)
    parser.add_argument('-d_ff', type=int, default=1024)
    parser.add_argument('-num_layers', type=int, default=6)
    parser.add_argument('-num_classes', type=int, default=2)

    parser.add_argument('-activation', type=str, default='gelu')
    parser.add_argument('-norm', type=str, default='BatchNorm')

    parser.add_argument('-seed', type=int, default=None)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-scale_factor', type=float, default='1.0')

    parser.add_argument('-print_interval', type=int, default=40)

    parser.add_argument('-n_gpu', type=int, default=2)

    parser.add_argument('-output_dir', type=str, default='./output')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    return parser.parse_args()
