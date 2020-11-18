import torch
import numpy as np
import copy
import json
import pickle
from tqdm import tqdm

from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict

from data_generation import generate_regular_learning_data


def generate_cf_data():
    dataset_path = "movielens/ml-1m/warm_state.json"
    dataset_path_y = "movielens/ml-1m/warm_state_y.json"
    data = []
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.loads(f.read())
    with open(dataset_path_y, encoding="utf-8") as f:
        dataset_y = json.loads(f.read())
    for _, user_id in tqdm(enumerate(dataset.keys())):
        tmp_x = np.array(dataset[user_id])
        tmp_y = np.array(dataset_y[user_id])
        for i in range(tmp_x.shape[0]):
            data.append([int(user_id), tmp_x[i], tmp_y[i]])
    # df = pd.DataFrame(list_dataset, columns =['user_id', 'item_id', 'rating'])
    data = np.array(data, dtype=int)
    np.random.shuffle(data)
    return data

class EmbeddingDot(torch.nn.Module):
    def __init__(self, n_users, n_movies, n_factors):
        super().__init__()
        self.u = torch.nn.Embedding(n_users, n_factors)
        self.m = torch.nn.Embedding(n_movies, n_factors)
        self.u.weight.data.uniform_(0,0.05)
        self.m.weight.data.uniform_(0,0.05)

    def forward(self, cats):
        users,movies = cats[:,0],cats[:,1]
        u,m = self.u(users),self.m(movies)
        return (u*m).sum(1).view((-1, 1))


def train_cf():
    n_factors = 64
    num_epoches = 50
    batch_size = 2048
    data = generate_regular_learning_data()
    data_size = data.shape[0]
    max_num = data.max(axis=0)
    n_users = max_num[0] + 1
    n_movies = max_num[1] + 1
    # train_set = torch.tensor(data[data_size // 5:])
    train_set = torch.tensor(data)
    val_test = torch.tensor(data[:data_size // 5])
    num_batches = train_set.shape[0] // batch_size
    print('num batches', num_batches)
    wd=1e-4
    model = EmbeddingDot(n_users, n_movies, n_factors)
    optim = torch.optim.Adam(model.parameters(), 1e-2, weight_decay=wd)
    for epoch in range(num_epoches):
        train_set = train_set[torch.randperm(train_set.shape[0])]
        pred = model(train_set[:,:2])
        train_loss = F.mse_loss(pred, train_set[:,2].view(-1, 1))
        pred = model(val_test[:,:2])
        val_loss = F.mse_loss(pred, val_test[:,2].view(-1, 1))
        print("CF epoch", epoch, "train loss", train_loss, "val loss", val_loss)
        for b in range(num_batches):
            train_batch = train_set[b*batch_size:b*batch_size+batch_size]
            pred = model(train_batch[:,:2])
            loss = F.mse_loss(pred, train_batch[:,2].type(torch.FloatTensor).view(-1, 1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            # if b % 20 == 0:
            #     print("CF epoch", epoch, "batch", b, "loss", loss)
    return model

model = train_cf()
item_emb = torch.tensor(model.m.weight, requires_grad=False)
print(item_emb[[1000, 2000, 3000]])

pickle.dump(item_emb, open("./ml/cf_item_embeddings.pkl", "wb"))

# item_emb = pickle.load(open("./ml/cf_item_embeddings.pkl", "rb"))
# print(item_emb[[1000, 2000, 3000]])
