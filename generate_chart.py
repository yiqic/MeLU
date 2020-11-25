import os
import torch
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np

from MeLU import MeLU
from options import config, states


def load_results():
    names = {
        'baseline_cf_False': '1. Baseline Model',
        'baseline_cf_True': '2. Baseline Model with ID-based Item Embedding',
        'melu_traink_100_learnlr_False_aug_False_cf_False_inner_1': '3. MAML in MeLU',
        'melu_traink_5_learnlr_False_aug_False_cf_False_inner_1': '4. MAML with Aligned Task Construction',
        # 'melu_traink_100_learnlr_True_aug_False_cf_False_inner_1': 'MAML with Learnable Inner LR',
        'melu_traink_5_learnlr_True_aug_False_cf_False_inner_1': '5. MAML in (4) + Learnable Inner LR',
        'melu_traink_5_learnlr_True_aug_True_cf_False_inner_1': '6. MAML in (5) + Training Set Augmentation',
        # 'melu_traink_5_learnlr_True_aug_False_cf_True_inner_1',
        'melu_traink_5_learnlr_True_aug_True_cf_True_inner_1': '7. MAML in (6) + ID-based Item Embedding',
        # 'melu_traink_100_learnlr_True_aug_False_cf_True_inner_1',
    }
    data = {}
    for name, legend in names.items():
        data[legend] = np.load('ml/result/{}_finetune80.npy'.format(name))
    return data


def generate_eval_curve_chart():
    data = load_results()
    x = np.arange(0, 80)
    for k, v in data.items():
        plt.plot(x, v[-1, 0:80], label=k)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.ylim((0, 8))
    plt.show()


def generate_optimal_setting_chart():
    data = load_results()['7. MAML in (6) + ID-based Item Embedding']
    max_epoch = 240
    x = np.arange(0, max_epoch)
    plt.plot(x, data[0, 0:max_epoch], label='train MSE loss pre local update')
    plt.plot(x, data[1, 0:max_epoch], label='train MSE loss post local update')
    plt.plot(x, data[2, 0:max_epoch], label='eval MSE loss pre local update')
    plt.plot(x, data[3, 0:max_epoch], label='eval MSE loss post local update')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.ylim((0, 2))
    plt.show()


def print_best_result():
    data = load_results()
    # best_res = {}
    baseline_res = np.min(data['1. Baseline Model'][-1])
    for k, v in data.items():
        min_idx = np.argmin(v[-1])
        # best_res[k] = np.min(v[-1])
        print(k, ':', v[:, min_idx], (v[-1, min_idx] - baseline_res) / baseline_res * 100)
    # print(best_res)


# generate_eval_curve_chart()
generate_optimal_setting_chart()
# print_best_result()
