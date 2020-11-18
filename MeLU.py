import torch
import numpy as np
import copy

from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict

from embeddings import item, user

def initialize_2d_weight(in_dim, out_dim):
    w = torch.empty(out_dim, in_dim)
    torch.nn.init.xavier_uniform_(w)
    # return Variable(w, requires_grad=True)
    return torch.nn.Parameter(w)

class user_preference_estimator(torch.nn.Module):

    def __init__(self, config):
        super(user_preference_estimator, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.include_item_embeddings = config['include_item_embeddings']
        self.fc1_in_dim = config['embedding_dim'] * (10 if config['include_item_embeddings'] else 8)
        self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']
        self.use_cuda = config['use_cuda']

        self.weights = OrderedDict()
        self.weights['fc1_weight'] = initialize_2d_weight(self.fc1_in_dim, self.fc2_in_dim)
        self.weights['fc1_bias'] = initialize_2d_weight(self.fc2_in_dim, 1)
        self.weights['fc2_weight'] = initialize_2d_weight(self.fc2_in_dim, self.fc2_out_dim)
        self.weights['fc2_bias'] = initialize_2d_weight(self.fc2_out_dim, 1)
        self.weights['fc3_weight'] = initialize_2d_weight(self.fc2_out_dim, 1)
        self.weights['fc3_bias'] = initialize_2d_weight(1, 1)
        for k, v in self.weights.items():
            self.register_parameter(k, v)

        self.item_emb = item(config)
        self.user_emb = user(config)
        # self.fc1 = torch.nn.Linear(self.fc1_in_dim, self.fc2_in_dim)
        # self.fc2 = torch.nn.Linear(self.fc2_in_dim, self.fc2_out_dim)
        # self.linear_out = torch.nn.Linear(self.fc2_out_dim, 1)

    def forward(self, x, item_cf_emb, weights = None, training = True):
        rate_idx = Variable(x[:, 0], requires_grad=False)
        genre_idx = Variable(x[:, 1:26], requires_grad=False)
        director_idx = Variable(x[:, 26:2212], requires_grad=False)
        actor_idx = Variable(x[:, 2212:10242], requires_grad=False)
        gender_idx = Variable(x[:, 10242], requires_grad=False)
        age_idx = Variable(x[:, 10243], requires_grad=False)
        occupation_idx = Variable(x[:, 10244], requires_grad=False)
        area_idx = Variable(x[:, 10245], requires_grad=False)

        if weights is None:
            weights = self.weights

        item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
        user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx)
        x = torch.cat((item_emb, user_emb), 1)
        if self.include_item_embeddings:
            x = torch.cat((x, item_cf_emb), 1)
        x = F.linear(x, weights['fc1_weight'], bias=weights['fc1_bias'])
        x = F.relu(x)
        x = F.linear(x, weights['fc2_weight'], bias=weights['fc2_bias'])
        x = F.relu(x)
        return F.linear(x, weights['fc3_weight'], bias=weights['fc3_bias'])


class MeLU(torch.nn.Module):
    def __init__(self, config):
        super(MeLU, self).__init__()
        self.config = config
        self.use_cuda = config['use_cuda']
        self.model = user_preference_estimator(config)
        self.local_lr = config['local_lr']
        self.num_local_update = config['inner']
        # self.local_update_target_weight_name = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'linear_out.weight', 'linear_out.bias']
        self.local_update_target_weight_name = self.model.weights.keys()
        # self.store_parameters()
        self.local_lr_vars = None
        if config['learn_local_lr']:
            self.local_lr_vars = OrderedDict()
            for weight in self.model.weights.keys():
                for idx in range(self.num_local_update):
                    self.local_lr_vars[weight + '_lr_' + str(idx)] = torch.nn.Parameter(torch.tensor(self.local_lr))
            for k, v in self.local_lr_vars.items():
                self.register_parameter(k, v)
            # self.meta_optim.add_param_group(self.local_lr_vars)
        self.meta_optim = torch.optim.Adam(self.parameters(), lr=config['lr'])
        print(self.state_dict().keys())


    def forward(self, support_set_x, support_set_y, query_set_x, support_set_cf = None, query_set_cf = None):
        query_set_y_pred_pre = self.model(query_set_x, item_cf_emb=query_set_cf)
        # cloned_model = copy.deepcopy(self.model)
        cur_weights = self.model.weights
        for idx in range(self.num_local_update):
            next_weights = OrderedDict()
            # weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.model(support_set_x, weights=cur_weights, item_cf_emb=support_set_cf)
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
            # self.model.zero_grad()
            grad = torch.autograd.grad(loss, cur_weights.values(), create_graph=True)
            # local update
            i = 0
            for k, v in cur_weights.items():
                # if self.weight_name[i] in self.local_update_target_weight_name:
                if self.local_lr_vars is not None:
                    local_lr = self.local_lr_vars[k + '_lr_' + str(idx)]
                else:
                    local_lr = self.local_lr
                next_weights[k] = v - local_lr * grad[i]
                i += 1
            # self.local_update(cloned_model, fast_weights)
            cur_weights = next_weights

        query_set_y_pred = self.model(query_set_x, weights=cur_weights, item_cf_emb=query_set_cf)
        # self.local_update(self.keep_weight)
        return query_set_y_pred_pre, query_set_y_pred

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys, support_set_cfs = None, query_set_cfs = None, training=True):
        batch_sz = len(support_set_xs)
        losses_q = []
        losses_q_pre = []
        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()
        for i in range(batch_sz):
            query_set_y_pred_pre, query_set_y_pred = self.forward(
                support_set_xs[i],
                support_set_ys[i],
                query_set_xs[i],
                support_set_cfs[i] if support_set_cfs is not None else None,
                query_set_cfs[i] if query_set_cfs is not None else None)
            loss_q = F.mse_loss(query_set_y_pred, query_set_ys[i].view(-1, 1))
            losses_q.append(loss_q)
            loss_q_pre = F.mse_loss(query_set_y_pred_pre, query_set_ys[i].view(-1, 1))
            losses_q_pre.append(loss_q_pre)
        losses_q = torch.stack(losses_q).mean(0)
        losses_q_pre = torch.stack(losses_q_pre).mean(0)
        # print('param2', self.model._parameters)
        # print('lr1', self.local_lr_vars['fc1_weight_lr_0'])
        # print('lr2', self.local_lr_vars['fc1_weight_lr_1'])
        if training:
            self.meta_optim.zero_grad()
            # print('before1', self.model.state_dict()['user_emb.embedding_age.weight'])
            # print('before2', self.model.weights['fc1_bias'])
            # for param in self.model.parameters():
            #     print(param)
            losses_q.backward()
            # for param in self.model.parameters():
            #     print(param)
            self.meta_optim.step()
            # print('after1', self.model.state_dict()['user_emb.embedding_age.weight'])
            # print('after2', self.model.weights['fc1_bias'])
        # self.store_parameters()
        return losses_q_pre.data.item(), losses_q.data.item()

    def get_weight_avg_norm(self, support_set_x, support_set_y):
        tmp = 0.
        if self.cuda():
            support_set_x = support_set_x.cuda()
            support_set_y = support_set_y.cuda()
        for idx in range(self.num_local_update):
            if idx > 0:
                self.model.load_state_dict(self.fast_weights)
            weight_for_local_update = list(self.model.state_dict().values())
            support_set_y_pred = self.model(support_set_x)
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
            # unit loss
            loss /= torch.norm(loss).tolist()
            self.model.zero_grad()
            grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            for i in range(self.weight_len):
                # For averaging Forbenius norm.
                tmp += torch.norm(grad[i])
                if self.weight_name[i] in self.local_update_target_weight_name:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i] - self.local_lr * grad[i]
                else:
                    self.fast_weights[self.weight_name[i]] = weight_for_local_update[i]
        return tmp / self.num_local_update




    # def store_parameters(self):
    #     self.keep_weight = OrderedDict()
    #     for name in self.local_update_target_weight_name:
    #         layer, term = name.split('.')
    #         self.keep_weight[name] = getattr(getattr(self.model, layer), term)
    #         # if name == 'fc1.bias':
    #         #     print(self.keep_weight[name])
    #     self.weight_name = list(self.model.state_dict().keys())
    #     self.weight_len = len(self.weight_name)

    # def clone_model(self):
    #     new_model = user_preference_estimator(self.config)
    #     for name in self.weight_name:
    #         src_attr = self.model
    #         dst_attr = new_model
    #         terms = name.split('.')
    #         for i in range(len(terms) - 1):
    #             src_attr = getattr(src_attr, terms[i])
    #             dst_attr = getattr(dst_attr, terms[i])
    #         setattr(dst_attr, terms[-1], torch.nn.Parameter(getattr(src_attr, terms[-1]).clone()))
    #     return new_model

    # def local_update(self, model, weights):
    #     for name in self.local_update_target_weight_name:
    #         layer, term = name.split('.')
    #         setattr(getattr(model, layer), term, weights[name])
