<!-- .. MOGLAM documentation master file, created by
   sphinx-quickstart on Thu May 25 09:49:25 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive. -->

Welcome to MOGLAM's documentation!
==================================

MOGLAM is an end-to-end interpretable multi-omics integration method, which mainly consists of three modules: dynamic graph convolutional network with feature selection (FSDGCN), multi-omics attention mechanism (MOAM), and omic-integrated representation learning (OIRL).


Tutorial
==================

In this tutorial, we show how to apply MOGLAM to classify diseases and identify important biomarkers by integrating multi-omics data. The MOGLAM folder mainly contains *main_MOGLAM.py*, *train_test.py*, *models.py*, *layers.py*, *utils.py* and *param.py*, where *main_MOGLAM.py* is the main function that only needs to be run. 

Before running the MOGLAM method, please download the input data via <https://github.com/Ouyang-Dong/MOGLAM/tree/master/BRCA_split>

## Runing the model
Through only running the *main_MOGLAM.py* file, we can train the MOGLAM model to output the values of accuracy (ACC), average F1 score weighted by support (F1_weighted) and macro-averaged F1 score (F1_macro).


	from train_test import train_test

	if __name__ == "__main__":
		data_folder = './BRCA_split/BRCA'

		view_list = [1,2,3]
		num_epoch_pretrain = 500
		num_epoch = 3000

		theta_smooth = 1
		theta_degree = 0.5
		theta_sparsity = 0.5

		if data_folder == './BRCA_split/BRCA':
			num_class = 5
			lr_e_pretrain = 1e-4
			lr_e = 1e-5
			lr_c = 1e-6
			reg = 0.001
			neta = 0.1

		train_test(data_folder, view_list, num_class,
				   lr_e_pretrain, lr_e, lr_c, 
				   num_epoch_pretrain, num_epoch, theta_smooth, theta_degree, theta_sparsity, neta, reg)


To make the reader a deeper understanding of our proposed MOGLAM, we mainly introduce the `train_test` module called in the *main_MOGLAM.py* file, and the `models` module mainly called in the *train_test.py* file in detail step-by-step. The detailed introduction is as follows.

### 1. Introduction to train_test.py
First, let's introduce the *train_test.py* file. The *train_test.py* mainly contains `train_test` ,`train_epoch`,`test_epoch` and `prepare_trte_data` functions. We can run `train_test` function to call `prepare_trte_data` function for reading training and test datasets, `train_epoch` function for training model, and `test_epoch` function for testing model.
#### 1.1 Reading training and test datasets
We utilize `np.loadtxt` to read training and test datasets, as well as their corresponding patient labels. At the same time, we use `torch.FloatTensor` to convert datasets into tensors.

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

#### 1.2 Calculating the initial patient similarity matrix
We make use of cosine similarity to calculate the initial patient similarity matrix for each omics.

	def gen_trte_adj_mat(data_tr_list, data_te_list, adj_parameter):
		adj_metric = "cosine"
		adj_train_list = []
		adj_test_list = []
		for i in range(len(data_tr_list)):
			adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
			adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
			adj_test_list.append(gen_adj_mat_tensor(data_te_list[i], adj_parameter_adaptive, adj_metric))
		
		return adj_train_list, adj_test_list



#### 1.3 Training model
We first update the dynamic graph convolutional network with feature selection (FSDGCN) by defining inner product regularization, cross-entropy and graph structure learning loss. Then, when *train_MOAM_OIRL = True*, the model starts to update remaining two modules, namely, multi-omics attention mechanism (MOAM), and omic-integrated
representation learning (OIRL).

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

#### 1.4 Testing model
After the MOGLAM model is trained, we use the defined `test_epoch` function to test the prediction performance of the model on the test dataset.

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

### 2. Introduction to models.py
In the *models.py* file, `GraphLearn` class is used for adaptive graph learning, `GCN_E` class is used to define graph convolutional networks, `Multiomics_Attention_mechanism` class is used to define multi-omics attention mechanism and `TransformerEncoder` class is used to define omic-integrated representation learning.

#### 2.1 Adaptive graph learning
In the proposed MOGLAM method, we use weighted cosine similarity (i.e., self.mode == 'weighted-cosine') for graph structure learning, which enables the model to achieve better classification performance. Finally, we can obtain the final patient similarity matrix for each omics by integrating the initial patient similarity and the learned patient similarity obtained by adaptive graph learning.

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

#### 2.2 Graph convolutional network
We define a two-layer graph convolutional network and only multiply the weight matrix (W_s) and the feature matrix (X) in the first layer (i.e., flag=True) to achieve dimensionality reduction.

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

#### 2.3 Multi-omics attention mechanism
We first obtain the initial attention for each omics using `nn.AdaptiveAvgPool2d` function. Then, we input the calculated initial attention into a two-layer fully connected network including Relu and Sigmoid activation for multi-omics attention learning.

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

#### 2.4 Omic-integrated representation learning
We apply `FeedForwardLayer` and `EncodeLayer` to define feedforward network and multi-head self-attention for capturing common and complementary information, respectively.

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



