
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from param import parameter_parser
import torch.nn.functional as F
from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, cal_adj_mat_parameter
from utils import GraphConstructLoss, normalize_adj


cuda = True if torch.cuda.is_available() else False

def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]

    data_tr_tensor_list = []
    data_te_tensor_list = []
    for i in range(num_view):
        data_tr_tensor_list.append(torch.FloatTensor(data_tr_list[i]))
        data_te_tensor_list.append(torch.FloatTensor(data_te_list[i]))
        if cuda:
            data_tr_tensor_list[i] = data_tr_tensor_list[i].cuda()
            data_te_tensor_list[i] = data_te_tensor_list[i].cuda()

    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))

    labels = np.concatenate((labels_tr, labels_te))
    
    return data_tr_tensor_list, data_te_tensor_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_te_list, adj_parameter):
    adj_metric = "cosine"
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_adj_mat_tensor(data_te_list[i], adj_parameter_adaptive, adj_metric))
    
    return adj_train_list, adj_test_list


def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, theta_smooth, theta_degree, theta_sparsity, neta, train_MOAM_OIRL=True):
    loss_dict = {}
    for m in model_dict:
        model_dict[m].train()

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    weight1 = list(model_dict['E1'].parameters())[0]
    weight2 = list(model_dict['E2'].parameters())[0]
    weight3 = list(model_dict['E3'].parameters())[0]
    WF_weight = [weight1, weight2, weight3]

    num_view = len(data_list)
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)].zero_grad()
        ci_loss = 0

        adj_train = model_dict["GL{:}".format(i + 1)](data_list[i])
        graph_loss = GraphConstructLoss(data_list[i], adj_train, adj_list[i], theta_smooth, theta_degree, theta_sparsity)
        final_adj = neta * adj_train + (1-neta) * adj_list[i]
        normalized_adj = normalize_adj(final_adj)

        ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],normalized_adj))
        ci_loss = torch.mean(torch.mul(criterion(ci,label), sample_weight))

        '''inner product regularization'''
        new_WF_weight = torch.mm(WF_weight[i], WF_weight[i].T)
        WF_L1_list = torch.norm(new_WF_weight, p=1)
        WF_L2_list = torch.pow(torch.norm(WF_weight[i], p=2), 2)
        WF_L12_loss = WF_L1_list - WF_L2_list

        tol_loss = ci_loss + graph_loss + 0.0001 * WF_L12_loss
        tol_loss.backward()
        optim_dict["C{:}".format(i+1)].step()
        loss_dict["C{:}".format(i+1)] = tol_loss.detach().cpu().numpy().item()

    if train_MOAM_OIRL and num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        GCN_list = []
        for i in range(num_view):
            adj_train = model_dict["GL{:}".format(i + 1)](data_list[i])
            final_adj = neta * adj_train + (1 - neta) * adj_list[i]
            normalized_adj = normalize_adj(final_adj)

            GCN_list.append(model_dict["E{:}".format(i+1)](data_list[i],normalized_adj))

        atten_data_list = model_dict["MOAM"](GCN_list)
        new_data = torch.cat([atten_data_list[0],atten_data_list[1],atten_data_list[2]],dim= 1)
        c = model_dict["OIRL"](new_data)
        c_loss = torch.mean(torch.mul(criterion(c, label), sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()
    
    return loss_dict
    

def test_epoch(data_list, adj_list, model_dict, neta):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        adj_test = model_dict["GL{:}".format(i + 1)](data_list[i])
        final_adj = neta * adj_test + (1 - neta) * adj_list[i]
        normalized_adj = normalize_adj(final_adj)

        ci_list.append(model_dict["E{:}".format(i+1)](data_list[i],normalized_adj))

    atten_data_list = model_dict["MOAM"](ci_list)
    new_data = torch.cat([atten_data_list[0], atten_data_list[1], atten_data_list[2]], dim=1)
    if num_view >= 2:
        c = model_dict["OIRL"](new_data)
    else:
        c = ci_list[0]

    prob = F.softmax(c, dim=1).data.cpu().numpy()
    
    return prob


def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch, theta_smooth, theta_degree, theta_sparsity, neta, reg):

    test_inverval = 50
    num_view = len(view_list)

    if data_folder == './BRCA_split/BRCA':
        adj_parameter = 8
        mode = 'weighted-cosine'
        featuresSelect_list = [400, 400, 400]
        dim_he_list = [400,400]
        input_data_dim = [dim_he_list[-1], dim_he_list[-1], dim_he_list[-1]]

    args = parameter_parser()

    data_tr_list, data_te_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class,use_sample_weight = True)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_te_list, adj_parameter)

    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, input_data_dim, args, adj_parameter, mode, featuresSelect_list)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    
    print("\nPretrain FSDGCNs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c, reg)
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, theta_smooth, theta_degree, theta_sparsity, neta, train_MOAM_OIRL=False)

    print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c,reg)
    for epoch in range(num_epoch+1):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, theta_smooth, theta_degree, theta_sparsity, neta)
        if epoch % test_inverval == 0:
            te_prob = test_epoch(data_te_list, adj_te_list, model_dict, neta)
            print("\nTest: Epoch {:d}".format(epoch))
           
            print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            print("Test F1 weighted: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
            print("Test F1 macro: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))