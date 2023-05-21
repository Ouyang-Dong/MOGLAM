""" Componets of the model
"""
import torch.nn as nn
import torch
import random
import os
import numpy as np
import torch.nn.functional as F
from layers import VariLengthInputLayer, EncodeLayer, FeedForwardLayer, OutputLayer
from utils import gen_adj_mat_tensor, cal_adj_mat_parameter

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_torch(seed=1234)


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)


class GraphLearn(nn.Module):
    def __init__(self, input_dim, adj_parameter, mode):
        super(GraphLearn, self).__init__()
        self.mode = mode
        self.w = nn.Sequential(nn.Linear(input_dim, 1))
        self.p = nn.Sequential(nn.Linear(input_dim, input_dim))

        self.w.apply(xavier_init)
        self.p.apply(xavier_init)

        self.adj_metric = "cosine"  # cosine distance
        self.adj_parameter = adj_parameter


    def forward(self,x):
        initial_x = x.clone()
        num, feat_dim = x.size(0), x.size(1)

        if self.mode == 'adaptive-learning':
            x = x.repeat_interleave(num, dim=0)
            x = x.view(num, num, feat_dim)
            diff = abs(x - initial_x)
            diff = F.relu(self.w(diff)).view(num, num)
            output = F.softmax(diff, dim=1)

        elif self.mode == 'weighted-cosine':
            x = self.p(x)
            x_norm = F.normalize(x, dim=-1)
            adj_parameter_adaptive = cal_adj_mat_parameter(self.adj_parameter, x_norm, self.adj_metric)
            output = gen_adj_mat_tensor(x_norm, adj_parameter_adaptive, self.adj_metric)

        return output


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, featuresSelected, flag, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.featuresSelect = featuresSelected

        if flag:
            self.WF = nn.Parameter(torch.FloatTensor(in_features, self.featuresSelect),requires_grad=True)
            self.weight = nn.Parameter(torch.FloatTensor(self.featuresSelect, out_features))

            nn.init.xavier_normal_(self.WF.data)
            nn.init.xavier_normal_(self.weight.data)

        else:
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
            nn.init.xavier_normal_(self.weight.data)

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))

        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj, flag):

        if flag:
            temp_support = torch.mm(x, self.WF)
            support = torch.mm(temp_support, self.weight)
            output = torch.mm(adj, support)
        else:
            support = torch.mm(x, self.weight)
            output = torch.mm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim, featuresSelect, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim[0], featuresSelect, flag=True)
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1], featuresSelect, flag=False)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj, flag=True)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, flag=False)
        x = F.leaky_relu(x, 0.25)

        return x


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class Multiomics_Attention_mechanism(nn.Module):
    def __init__(self):
        super().__init__()

        self.hiddim = 3
        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_x1 = nn.Linear(in_features=3, out_features=self.hiddim)
        self.fc_x2 = nn.Linear(in_features=self.hiddim, out_features=3)
        self.sigmoidx = nn.Sigmoid()

    def forward(self,input_list):
        new_input_list1 = input_list[0].reshape(1, 1, input_list[0].shape[0], -1)
        new_input_list2 = input_list[1].reshape(1, 1, input_list[1].shape[0], -1)
        new_input_list3 = input_list[2].reshape(1, 1, input_list[2].shape[0], -1)
        XM = torch.cat((new_input_list1, new_input_list2, new_input_list3), 1)

        x_channel_attenttion = self.globalAvgPool(XM)

        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), -1)
        x_channel_attenttion = self.fc_x1(x_channel_attenttion)
        x_channel_attenttion = torch.relu(x_channel_attenttion)
        x_channel_attenttion = self.fc_x2(x_channel_attenttion)
        x_channel_attenttion = self.sigmoidx(x_channel_attenttion)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1, 1)

        XM_channel_attention = x_channel_attenttion * XM
        XM_channel_attention = torch.relu(XM_channel_attention)

        return XM_channel_attention[0]


class TransformerEncoder(nn.Module):
    def __init__(self, input_data_dims, hyperpm, num_class):
        super(TransformerEncoder, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm.n_hidden
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.n_layer = hyperpm.nlayer
        self.modal_num = hyperpm.nmodal
        self.n_class = num_class
        self.d_out = self.d_v * self.n_head * self.modal_num

        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)

        self.Encoder = []
        self.FeedForward = []

        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)

            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)

        d_in = self.d_v * self.n_head * self.modal_num
        self.Outputlayer = OutputLayer(d_in, self.d_v * self.n_head, self.n_class, self.modal_num, self.dropout)

    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        x, _attn = self.InputLayer(x)
        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())

        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn = _attn.mean(dim=1)
            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())

        x = x.view(bs, -1)
        attn_embedding = attn.view(bs, -1)
        output = self.Outputlayer(x, attn_embedding)
        return output

    
def init_model_dict(num_view, num_class, dim_list, dim_he_list, input_data_dims, hyperpm, adj_parameter, mode,featuresSelect_list,gcn_dopout=0.5):
    model_dict = {}

    for i in range(num_view):
        model_dict['GL{:}'.format(i+1)] = GraphLearn(dim_list[i],adj_parameter, mode)
        model_dict["E{:}".format(i+1)] = GCN_E(dim_list[i], dim_he_list, featuresSelect_list[i], gcn_dopout)
        model_dict["C{:}".format(i+1)] = LinearLayer(dim_he_list[-1], num_class)

    if num_view >= 2:
        model_dict["MOAM"] = Multiomics_Attention_mechanism()
        model_dict["OIRL"] = TransformerEncoder(input_data_dims, hyperpm, num_class)
    return model_dict


def init_optim(num_view, model_dict, lr_e, lr_c,reg):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(list(model_dict['GL{:}'.format(i+1)].parameters()) +
                list(model_dict["E{:}".format(i+1)].parameters()) + list(model_dict["C{:}".format(i+1)].parameters()),
                lr=lr_e, weight_decay=reg)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(list(model_dict["MOAM"].parameters()) + list(model_dict["OIRL"].parameters()), lr=lr_c,weight_decay=reg)
    return optim_dict