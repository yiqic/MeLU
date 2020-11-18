import os
import torch
import pickle
import random
import numpy as np

from MeLU import MeLU, user_preference_estimator
from options import config
from model_training import training
from data_generation import *
from torch.nn import functional as F
from evidence_candidate import selection


# if __name__ == "__main__":
#     master_path= "./ml"
#     train_k = 5
#     eval_k = 5
#     if not os.path.exists("{}/".format(master_path)):
#         os.mkdir("{}/".format(master_path))
#         # preparing dataset. It needs about 22GB of your hard disk space.
#         # generate(master_path)
#         generate_first_k(master_path, 5)
#         generate_first_k(master_path, 100)
#         generate_batch_k(master_path, 5)

#     # training model.
#     melu = MeLU(config)
#     model_filename = "{}/models.pkl".format(master_path)
#     if not os.path.exists(model_filename):
#         # Load training dataset.
#         training_set_size = int(len(os.listdir("{}/warm_state/batch".format(master_path))) / 4)
#         supp_xs_s = []
#         supp_ys_s = []
#         query_xs_s = []
#         query_ys_s = []
#         for idx in range(training_set_size):
#             supp_xs_s.append(pickle.load(open("{}/warm_state/batch/supp_x_size_{}_{}.pkl".format(master_path, train_k, idx), "rb")))
#             supp_ys_s.append(pickle.load(open("{}/warm_state/batch/supp_y_size_{}_{}.pkl".format(master_path, train_k, idx), "rb")))
#             query_xs_s.append(pickle.load(open("{}/warm_state/batch/query_x_size_{}_{}.pkl".format(master_path, train_k, idx), "rb")))
#             query_ys_s.append(pickle.load(open("{}/warm_state/batch/query_y_size_{}_{}.pkl".format(master_path, train_k, idx), "rb")))
#         total_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
#         del(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)
#         # Load eval dataset
#         eval_set_size = int(len(os.listdir("{}/user_cold_state".format(master_path))) / 8)
#         supp_xs_s = []
#         supp_ys_s = []
#         query_xs_s = []
#         query_ys_s = []
#         for idx in range(eval_set_size):
#             supp_xs_s.append(pickle.load(open("{}/user_cold_state/supp_x_first_{}_{}.pkl".format(master_path, eval_k, idx), "rb")))
#             supp_ys_s.append(pickle.load(open("{}/user_cold_state/supp_y_first_{}_{}.pkl".format(master_path, eval_k, idx), "rb")))
#             query_xs_s.append(pickle.load(open("{}/user_cold_state/query_x_first_{}_{}.pkl".format(master_path, eval_k, idx), "rb")))
#             query_ys_s.append(pickle.load(open("{}/user_cold_state/query_y_first_{}_{}.pkl".format(master_path, eval_k, idx), "rb")))
#         eval_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))
#         del(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)
#         training(melu, total_dataset, eval_dataset, batch_size=config['batch_size'], num_epoch=config['num_epoch'], model_save=False, model_filename=model_filename)
#     else:
#         trained_state_dict = torch.load(model_filename)
#         melu.load_state_dict(trained_state_dict)

    # selecting evidence candidates.
    # evidence_candidate_list = selection(melu, master_path, config['num_candidate'])
    # for movie, score in evidence_candidate_list:
    #     print(movie, score)




def run_melu():
    master_path= "./ml"
    train_k = config['train_k']
    eval_k = 5
    # generate_first_k_np(master_path, 100)
    # generate_first_k_np(master_path, 5)
    # generate_batch_k_np(master_path, 5)

    # training model.
    melu = MeLU(config)
    model_filename = "{}/models.pkl".format(master_path)
    if not os.path.exists(model_filename):
        user_dict = pickle.load(open("{}/user_content.pkl".format(master_path), "rb"))
        movie_dict = pickle.load(open("{}/movie_content.pkl".format(master_path), "rb"))
        movie_cf = pickle.load(open("{}/cf_item_embeddings.pkl".format(master_path), "rb"))
        # Load training dataset.
        raw_supp = pickle.load(
            open("{}/warm_state_supp_{}_{}.pkl".format(master_path, 'batch' if config['enable_data_aug'] else 'first', train_k), "rb")
        )
        raw_query = pickle.load(
            open("{}/warm_state_query_{}_{}.pkl".format(master_path, 'batch' if config['enable_data_aug'] else 'first', train_k), "rb")
        )
        training_set_size = len(raw_supp)
        print("training_set_size", training_set_size)
        supp_xs_s = []
        supp_ys_s = []
        supp_cf_s = []
        query_xs_s = []
        query_ys_s = []
        query_cf_s = []
        for idx in range(training_set_size):
            supp_u_ids = raw_supp[idx][:, 0]
            supp_m_ids = raw_supp[idx][:, 1]
            supp_ratings = raw_supp[idx][:, 2]
            supp_xs_s.append(torch.cat((movie_dict[supp_m_ids], user_dict[supp_u_ids]), 1))
            supp_ys_s.append(torch.FloatTensor(supp_ratings))
            supp_cf_s.append(movie_cf[supp_m_ids])

            query_u_ids = raw_query[idx][:, 0]
            query_m_ids = raw_query[idx][:, 1]
            query_ratings = raw_query[idx][:, 2]
            query_xs_s.append(torch.cat((movie_dict[query_m_ids], user_dict[query_u_ids]), 1))
            query_ys_s.append(torch.FloatTensor(query_ratings))
            query_cf_s.append(movie_cf[query_m_ids])
            # print('finish itr', idx)
        # print('hello2')
        total_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s, supp_cf_s, query_cf_s))
        del(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s, supp_cf_s, query_cf_s)
        print("done loading training")
        # Load eval dataset
        raw_supp = pickle.load(open("{}/user_cold_state_supp_first_{}.pkl".format(master_path, eval_k), "rb"))
        raw_query = pickle.load(open("{}/user_cold_state_query_first_{}.pkl".format(master_path, eval_k), "rb"))
        eval_set_size = len(raw_supp)
        supp_xs_s = []
        supp_ys_s = []
        supp_cf_s = []
        query_xs_s = []
        query_ys_s = []
        query_cf_s = []
        for idx in range(eval_set_size):
            supp_u_ids = raw_supp[idx][:, 0]
            supp_m_ids = raw_supp[idx][:, 1]
            supp_ratings = raw_supp[idx][:, 2]
            supp_xs_s.append(torch.cat((movie_dict[supp_m_ids], user_dict[supp_u_ids]), 1))
            supp_ys_s.append(torch.FloatTensor(supp_ratings))
            supp_cf_s.append(movie_cf[supp_m_ids])

            query_u_ids = raw_query[idx][:, 0]
            query_m_ids = raw_query[idx][:, 1]
            query_ratings = raw_query[idx][:, 2]
            query_xs_s.append(torch.cat((movie_dict[query_m_ids], user_dict[query_u_ids]), 1))
            query_ys_s.append(torch.FloatTensor(query_ratings))
            query_cf_s.append(movie_cf[query_m_ids])
        eval_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s, supp_cf_s, query_cf_s))
        del(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s, supp_cf_s, query_cf_s)
        print("done loading eval")
        training(melu, total_dataset, eval_dataset, batch_size=config['batch_size'], num_epoch=config['num_epoch'], model_save=False, model_filename=model_filename)


def run_baseline():
    master_path= "./ml"
    # generate_first_k_np(master_path, 5)
    batch_size = config['batch_size'] * 10
    num_epoches = 20

    user_dict = pickle.load(open("{}/user_content.pkl".format(master_path), "rb"))
    movie_dict = pickle.load(open("{}/movie_content.pkl".format(master_path), "rb"))
    movie_cf = pickle.load(open("{}/cf_item_embeddings.pkl".format(master_path), "rb"))

    train_data = generate_regular_learning_data()
    # u_ids = train_data[:, 0]
    # m_ids = train_data[:, 1]
    # ratings = train_data[:, 2]
    # train_xs = torch.cat((movie_dict[m_ids], user_dict[u_ids]), 1)
    # train_ys = torch.FloatTensor(ratings)
    # train_cfs = movie_cf[m_ids]
    train_size = train_data.shape[0]
    print('size', train_size)

    test_data = np.array(pickle.load(open("{}/user_cold_state_query_first_{}.pkl".format(master_path, 5), "rb"))).reshape((-1, 3))
    u_ids = test_data[:, 0]
    m_ids = test_data[:, 1]
    ratings = test_data[:, 2]
    test_xs = torch.cat((movie_dict[m_ids], user_dict[u_ids]), 1)
    test_ys = torch.FloatTensor(ratings)
    test_cfs = movie_cf[m_ids]

    num_batches = train_size // batch_size
    batches_per_val = num_batches // config['tests_per_epoch']
    model = user_preference_estimator(config)
    wd = 1e-5
    optim = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=wd)
    train_losses = []
    val_losses = []
    for epoch in range(config['num_epoch']):
        np.random.shuffle(train_data)
        # perm = torch.randperm(train_size)
        # train_xs = train_xs[perm]
        # train_ys = train_ys[perm]
        # train_cfs = train_cfs[perm]
        # pred = model(train_xs, train_cfs)
        # train_loss = F.mse_loss(pred, train_ys.view(-1, 1))
        pred = model(test_xs, test_cfs)
        val_loss = F.mse_loss(pred, test_ys.view(-1, 1))
        print("CF epoch", epoch, "val loss", val_loss)
        # print("CF epoch", epoch, "train loss", train_loss, "val loss", val_loss)
        for b in range(num_batches):
            # train_x = train_xs[b*batch_size:b*batch_size+batch_size]
            # train_y = train_ys[b*batch_size:b*batch_size+batch_size]
            # train_cf = train_cfs[b*batch_size:b*batch_size+batch_size]

            train_batch = train_data[b*batch_size:b*batch_size+batch_size]
            u_ids = train_batch[:, 0]
            m_ids = train_batch[:, 1]
            ratings = train_batch[:, 2]
            train_xs = torch.cat((movie_dict[m_ids], user_dict[u_ids]), 1)
            train_ys = torch.FloatTensor(ratings)
            train_cfs = movie_cf[m_ids]
            pred = model(train_xs, train_cfs)
            loss = F.mse_loss(pred, train_ys.view(-1, 1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            if b % batches_per_val == 0 and b < batches_per_val * config['tests_per_epoch']:
                train_losses.append(loss.item())
                pred = model(test_xs, test_cfs)
                val_loss = F.mse_loss(pred, test_ys.view(-1, 1))
                print("epoch", epoch, "batch", b, "train loss", loss.item(), "val loss", val_loss.item())
                val_losses.append(val_loss.item())

    losses = np.array([train_losses, val_losses])
    print('logged data shape', losses.shape)
    np.save(
        "{}/result/baseline_cf_{}".format(master_path, config["include_item_embeddings"]),
        losses
    )



if __name__ == "__main__":
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    # run_baseline()
    run_melu()
