import os
import torch
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np

from MeLU import MeLU
from options import config, states


def training(melu, total_dataset, eval_dataset, batch_size, num_epoch, model_save=True, model_filename=None):
    if config['use_cuda']:
        melu.cuda()

    training_set_size = len(total_dataset)
    print('training set size', training_set_size)
    print('eval set size', len(eval_dataset))
    melu.train()
    train_pre_losses = []
    train_losses = []
    eval_pre_losses = []
    eval_losses = []
    for epoch in range(num_epoch):
        # supp_xs,supp_ys,query_xs,query_ys,supp_cfs,query_cfs = zip(*eval_dataset)
        # pre_loss, loss = melu.global_update(supp_xs, supp_ys, query_xs, query_ys, supp_cfs, query_cfs, training=False)
        # eval_pre_losses.append(pre_loss)
        # eval_losses.append(loss)
        # print('test result', epo, pre_loss, loss)
        random.shuffle(total_dataset)
        num_batch = int(training_set_size / batch_size)
        batches_per_val = num_batch // config['tests_per_epoch']
        a,b,c,d,acf,bcf = zip(*total_dataset)
        # pre_loss, loss = melu.global_update(a[:3200], b[:3200], c[:3200], d[:3200], acf[:3200], bcf[:3200], training=False)
        # print('train result', epo, pre_loss, loss)
        # train_pre_losses.append(pre_loss)
        # train_losses.append(loss)
        for i in range(num_batch):
            try:
                supp_xs = list(a[batch_size*i:batch_size*(i+1)])
                supp_ys = list(b[batch_size*i:batch_size*(i+1)])
                query_xs = list(c[batch_size*i:batch_size*(i+1)])
                query_ys = list(d[batch_size*i:batch_size*(i+1)])
                supp_cfs = list(acf[batch_size*i:batch_size*(i+1)])
                query_cfs = list(bcf[batch_size*i:batch_size*(i+1)])
            except IndexError:
                continue
            pre_loss, loss = melu.global_update(supp_xs, supp_ys, query_xs, query_ys, supp_cfs, query_cfs)
            if i % batches_per_val == 0 and i < batches_per_val * config['tests_per_epoch']:
                train_pre_losses.append(pre_loss)
                train_losses.append(loss)
                supp_xs,supp_ys,query_xs,query_ys,supp_cfs,query_cfs = zip(*eval_dataset)
                eval_pre_loss, eval_loss = melu.global_update(supp_xs, supp_ys, query_xs, query_ys, supp_cfs, query_cfs, training=False)
                eval_pre_losses.append(eval_pre_loss)
                eval_losses.append(eval_loss)
                print("epoch", epoch, "batch", i, "train pre loss", pre_loss, "post loss", loss, "val pre loss", eval_pre_loss, "post loss", eval_loss)

    # x = range(num_epoch)
    # plt.plot(x, train_pre_losses, label='train (before train on task)')
    # plt.plot(x, train_losses, label='train (after train on task)')
    # plt.plot(x, eval_pre_losses, label='eval (before train on task)')
    # plt.plot(x, eval_losses, label='eval (after train on task)')
    # plt.xlabel('epoch')
    # plt.ylabel('MSE loss')
    # plt.legend()
    # plt.title('MSE loss for MeLU network')
    # plt.savefig('mse_melu_full.png')
    losses = np.array([train_pre_losses, train_losses, eval_pre_losses, eval_losses])
    print('logged data shape', losses.shape)
    master_path= "./ml"
    np.save(
        "{}/result/melu_traink_{}_learnlr_{}_aug_{}_cf_{}_inner_{}".format(
            master_path, config["train_k"], config["learn_local_lr"], config["enable_data_aug"], config["include_item_embeddings"], config["inner"]
        ),
        losses
    )

    if model_save:
        torch.save(melu.state_dict(), model_filename)
