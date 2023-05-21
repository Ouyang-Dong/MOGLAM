import os
import numpy as np
import torch


cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cal_sample_weight(labels, num_class, use_sample_weight=True):

    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)

    count_dict = dict()
    for i in range(num_class):
        count_dict["{:}".format(i)] = np.sum(labels==i)

    sort_count_dict_Asc = dict(sorted(count_dict.items(), key=lambda x: x[1]))
    sort_count_dict_Des = dict(sorted(count_dict.items(), key=lambda x: x[1],reverse=True))

    count = list(sort_count_dict_Asc.values())
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        num = int(list(sort_count_dict_Des.keys())[i])
        new_count = list(sort_count_dict_Asc.values())[i]
        sample_weight[np.where(labels==num)[0]] = new_count/np.sum(count)

    return sample_weight


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)
    
    return y_onehot


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    parameter = torch.sort(dist.reshape(-1,)).values[edge_per_node*data.shape[0]]
    return np.ndarray.item(parameter.data.cpu().numpy())

def graph_from_dist_tensor(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0]==dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0
        
    return g


def gen_adj_mat_tensor(data, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    if metric == "cosine":
        adj = 1-dist
    else:
        raise NotImplementedError
    adj = adj*g 
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = adj + I

    return adj

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    D = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(D, -0.5)
    d_inv_sqrt = torch.diagflat(d_inv_sqrt)
    adj = torch.mm(d_inv_sqrt, torch.mm(adj, d_inv_sqrt))
    return adj


def GraphConstructLoss(feat, adj_list, true_adj_list, theta_smooth, theta_degree, theta_sparsity):
    # Graph regularization
    L = torch.diagflat(torch.sum(adj_list, -1)) - adj_list

    smoothess_penalty = torch.trace(torch.mm(feat.T, torch.mm(L, feat))) / int(np.prod(adj_list.shape))
    degree_penalty = torch.sum(torch.pow((adj_list - true_adj_list),2)) / int(np.prod(adj_list.shape))
    sparsity_penalty = torch.sum(torch.pow(adj_list, 2)) / int(np.prod(adj_list.shape))

    return theta_smooth * smoothess_penalty + theta_degree * degree_penalty + theta_sparsity * sparsity_penalty


def save_model_dict(folder, model_dict):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for module in model_dict:
        torch.save(model_dict[module].state_dict(), os.path.join(folder, module+".pth"))
            
    
def load_model_dict(folder, model_dict):
    for module in model_dict:
        if os.path.exists(os.path.join(folder, module+".pth")):
#            print("Module {:} loaded!".format(module))
            model_dict[module].load_state_dict(torch.load(os.path.join(folder, module+".pth"), map_location="cuda:{:}".format(torch.cuda.current_device())))
        else:
            print("WARNING: Module {:} from model_dict is not loaded!".format(module))
        if cuda:
            model_dict[module].cuda()    
    return model_dict
