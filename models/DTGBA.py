#%%
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from models.GCN import GCN
from models.losses import compute_gan_loss,percept_loss, compute_D_loss

#%%
class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input>thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None

class GraphTrojanNet(nn.Module):
    # In the furture, we may use a GNN model to generate backdoor
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers.append(nn.Linear(nfeat, nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        
        self.layers = nn.Sequential(*layers).to(device)

        self.feat = nn.Linear(nfeat,nout*nfeat)
        self.edge = nn.Linear(nfeat, int(nout*(nout-1)/2))
        self.device = device

    def forward(self, input, thrd):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """

        GW = GradWhere.apply
        self.layers = self.layers
        h = self.layers(input)

        feat = self.feat(h)
        edge_weight = self.edge(h)
        # feat = GW(feat, thrd, self.device)
        edge_weight = GW(edge_weight, thrd, self.device)

        return feat, edge_weight

class HomoLoss(nn.Module):
    def __init__(self,args,device):
        super(HomoLoss, self).__init__()
        self.args = args
        self.device = device
        
    def forward(self,trigger_edge_index,trigger_edge_weights,x,thrd):

        trigger_edge_index = trigger_edge_index[:,trigger_edge_weights>0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]],x[trigger_edge_index[1]])
        
        loss = torch.relu(thrd - edge_sims).mean()
        # print(edge_sims.min())
        return loss

#%%
import numpy as np
class Backdoor:

    def __init__(self,args, device):
        self.args = args
        self.device = device
        self.weights = None
        self.trigger_index = self.get_trigger_index(args.trigger_size)
        self.all_inject_edges = None
        self.first = True
        
    def get_trigger_index(self,trigger_size):
        edge_list = []
        edge_list.append([0,0])
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j,k])
        edge_index = torch.tensor(edge_list,device=self.device).long().T
        return edge_index

    def get_trojan_edge(self,start, idx_attach, trigger_size):
        edge_list = []
        for idx in idx_attach:
            edges = self.trigger_index.clone()
            edges[0,0] = idx
            edges[1,0] = start
            edges[:,1:] = edges[:,1:] + start

            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list,dim=1)
        # to undirected
        # row, col = edge_index
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1],edge_index[0]])
        edge_index = torch.stack([row,col])

        return edge_index
        
    def inject_trigger(self, idx_attach, features,edge_index,edge_weight,device):
        self.trojan = self.trojan.to(device)
        idx_attach = idx_attach.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        self.trojan.eval()

        outer_feature = features[idx_attach]
            
        vector_pos2 = torch.zeros(len(idx_attach), features.shape[1])
        vector_pos2[: , self.top_k_index_tensor] = 1.0

        vector_neg2 = torch.ones(len(idx_attach), features.shape[1])
        vector_neg2[: , self.least_k_index_tensor] = 0.0

        vector_pos2 = vector_pos2.to(self.device)
        vector_neg2 = vector_neg2.to(self.device)

        outer_feature = outer_feature * vector_neg2 + vector_pos2

        outer_feature = torch.clamp(outer_feature , min=0.0, max=1.0)

        trojan_feat, trojan_weights = self.trojan(features[idx_attach],self.args.thrd) # may revise the process of generate
        
        trojan_weights = torch.cat([torch.ones([len(idx_attach),1],dtype=torch.float,device=device),trojan_weights],dim=1)
        trojan_weights = trojan_weights.flatten()

        trojan_feat = trojan_feat.view([-1,features.shape[1]])

        trojan_edge = self.get_trojan_edge(len(features),idx_attach,self.args.trigger_size).to(device)

        update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
        update_feat = torch.cat([features,trojan_feat])
        update_feat[idx_attach] = outer_feature
        update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

        self.trojan = self.trojan.cpu()

        if self.first == True:
            self.first = False
            self.all_inject_edges = deepcopy(trojan_edge)
        else:
            self.all_inject_edges =  torch.cat([self.all_inject_edges, trojan_edge],dim=1)

        idx_attach = idx_attach.cpu()
        features = features.cpu()
        edge_index = edge_index.cpu()
        edge_weight = edge_weight.cpu()
        return update_feat, update_edge_index, update_edge_weights

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach,idx_unlabeled,  netD, optimizer_D, Dopt, rep_net ,loss_type='gan', vector_neg=None, vector_pos=None, top_k_index_tensor=None, least_k_index_tensor=None):
        ori_n = features.shape[0]
        args = self.args
        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]],device=self.device,dtype=torch.float)
        self.idx_attach = idx_attach
        self.features = features
        self.edge_index = edge_index
        self.edge_weights = edge_weight
        self.top_k_index_tensor = top_k_index_tensor
        self.least_k_index_tensor = least_k_index_tensor
        mask_neg = vector_neg
        mask_pos = vector_pos
        # initial a shadow model
        self.shadow_model = GCN(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=labels.max().item() + 1,
                         dropout=0.0, device=self.device).to(self.device)
        # initalize a trojanNet to generate trigger
        self.trojan = GraphTrojanNet(self.device, features.shape[1], args.trigger_size, layernum=2).to(self.device)
        self.homo_loss = HomoLoss(self.args,self.device)

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # change the labels of the poisoned node to the target class
        self.labels = labels.clone()
        self.labels[idx_attach] = args.target_class
        # 使用掩码从tensor1中取出元素
        self.un_idx_attach = np.setdiff1d(idx_train.detach().cpu(), idx_attach.detach().cpu())

        # print('len(idx_attach):')
        # print(len(idx_attach))

        # get the trojan edges, which include the target-trigger edge and the edges among trigger
        trojan_edge = self.get_trojan_edge(len(features),idx_attach,args.trigger_size).to(self.device)

        # update the poisoned graph's edge index
        poison_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

        # furture change it to bilevel optimization
        bs = 80
        loss_best = 1e8

        Dopt_out_epoch = self.args.Dopt
        for i in range(args.trojan_epochs):
            self.trojan.train()
            for j in range(self.args.inner):
                optimizer_shadow.zero_grad()

                trojan_feat0 = features[idx_attach]
                # print("trojan_feat:", trojan_feat0)
                # print(trojan_feat0.sum())
                trojan_feat0 = trojan_feat0 * mask_neg + mask_pos
                # print(trojan_feat0.sum())
                trojan_feat0 = torch.clamp(trojan_feat0 , min=0.0, max=1.0)
                # print(trojan_feat0.sum())
                # print("trojan_feat0.shape:", trojan_feat0.shape)
                
                trojan_feat, trojan_weights = self.trojan(features[idx_attach],args.thrd) # may revise the process of generate
                trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                trojan_weights = trojan_weights.flatten()
                trojan_feat = trojan_feat.view([-1,features.shape[1]])

                poison_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
                poison_x = torch.cat([features,trojan_feat]).detach()
                poison_x[idx_attach] = trojan_feat0
                output = self.shadow_model(poison_x, poison_edge_index, poison_edge_weights)
                
                loss_inner = F.nll_loss(output[torch.cat([idx_train,idx_attach])], self.labels[torch.cat([idx_train,idx_attach])]) # add our adaptive loss
                
                loss_inner.backward()
                optimizer_shadow.step()
            
            new_n = ori_n + trojan_feat.shape[0]
            n = ori_n

            netD.train()
            for Dopt_ep in range(Dopt):
                if new_n - n > bs:
                    fake_batch =  np.random.randint(n, new_n, (bs))
                else:
                    fake_batch = np.arange(n,new_n)
                #n -> new_n 是fake nodes
                fake_batch = idx_attach
                #0 -> n-1 是real nodes，随机取等量的real nodes，让判别器判断哪些是real？哪些是fake？

                indices = np.random.permutation(len(self.un_idx_attach))
                # real_batch =  self.un_idx_attach[indices[:len(fake_batch)]]
                real_batch = fake_batch

                # print('current info')
                # print('features:')
                # print(features.shape)
                # print(type(features))
                # print('trojan_feat:')
                # print(trojan_feat.shape)
                # print(type(trojan_feat))
                # print('edge_index:')
                # print(edge_index.shape)
                # print(type(edge_index))
                # print('trojan_edge:')
                # print(trojan_edge.shape)
                # print(type(trojan_edge))
                # print(real_batch)
                # print(fake_batch)

                #训练 判别器
                train_loss_D, train_acc_D, train_acc_real, train_acc_fake = compute_D_loss(real_batch, fake_batch, netD, edge_index, features, trojan_feat, trojan_edge,ori_n, new_n, self.device)
                optimizer_D.zero_grad()
                train_loss_D.backward()
                nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.1)
                optimizer_D.step()
                train_loss_D_detach = train_loss_D.detach().item()
                del train_loss_D
                # print('Acc D', train_acc_D)
                # print('Loss D', train_loss_D_detach)
                # print('Each opt D/acc real D', train_acc_real)
                # print('Each opt D/acc fake D', train_acc_fake)
                # print('Each opt D/acc D', train_acc_D)
                # print('Each opt D/loss D', train_loss_D_detach)
                Dopt_out_epoch += 1
            
            netD.eval()

            trojan_feat_tensor = torch.cat((features, trojan_feat), dim=0)
            trojan_adj =  torch.cat((edge_index, trojan_edge), dim=1)
            trojan_adj_tensor = tensor2sparse_coo_tensor(trojan_adj, new_n, self.device)

            # all_emb_list = rep_net(trojan_feat_tensor, trojan_adj_tensor)
            pred_fake_G = netD(trojan_feat_tensor, trojan_adj_tensor)[1]
            loss_G_fake = compute_gan_loss(loss_type, pred_fake_G[fake_batch])
            # loss_G_div = percept_loss(all_emb_list, fake_batch)

            acc_train_clean = utils.accuracy(output[idx_train], self.labels[idx_train])
            acc_train_attach = utils.accuracy(output[idx_attach], self.labels[idx_attach])
            
            # involve unlabeled nodes in outter optimization
            self.trojan.eval()
            optimizer_trigger.zero_grad()

            rs = np.random.RandomState(self.args.seed)
            idx_outter = torch.cat([idx_attach,idx_unlabeled[rs.choice(len(idx_unlabeled),size=512,replace=False)]])

            #给测试阶段的目标节点注入触发器
            outer_feature = features[idx_outter]
            
            vector_pos2 = torch.zeros(len(idx_outter), features.shape[1])
            vector_pos2[: , top_k_index_tensor] = 1.0

            vector_neg2 = torch.ones(len(idx_outter), features.shape[1])
            vector_neg2[: , least_k_index_tensor] = 0.0

            vector_pos2 = vector_pos2.to(self.device)
            vector_neg2 = vector_neg2.to(self.device)

            outer_feature = outer_feature * vector_neg2 + vector_pos2

            outer_feature = torch.clamp(outer_feature , min=0.0, max=1.0)

            trojan_feat, trojan_weights = self.trojan(features[idx_outter],self.args.thrd) # may revise the process of generate
        
            trojan_weights = torch.cat([torch.ones([len(idx_outter),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
            trojan_weights = trojan_weights.flatten()

            trojan_feat = trojan_feat.view([-1,features.shape[1]])

            trojan_edge = self.get_trojan_edge(len(features),idx_outter,self.args.trigger_size).to(self.device)

            update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
            update_feat = torch.cat([features,trojan_feat])
            update_feat[idx_outter] = outer_feature
            update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

            output = self.shadow_model(update_feat, update_edge_index, update_edge_weights)

            labels_outter = labels.clone()
            labels_outter[idx_outter] = args.target_class
            loss_target = self.args.target_loss_weight *F.nll_loss(output[torch.cat([idx_train,idx_outter])],
                                    labels_outter[torch.cat([idx_train,idx_outter])])
            loss_homo = 0.0

            if(self.args.homo_loss_weight > 0):
                loss_homo = self.homo_loss(trojan_edge[:,:int(trojan_edge.shape[1]/2)],\
                                            trojan_weights,\
                                            update_feat,\
                                            self.args.homo_boost_thrd)
            
            loss_outter = (loss_target + self.args.homo_loss_weight * loss_homo +
                           self.args.alpha * loss_G_fake)

            loss_outter.backward()
            optimizer_trigger.step()
            acc_train_outter =(output[idx_outter].argmax(dim=1)==args.target_class).float().mean()

            if loss_outter<loss_best:
                self.weights = deepcopy(self.trojan.state_dict())
                loss_best = float(loss_outter)

            if args.debug and i % 10 == 0:
                print('Epoch {}, loss_inner: {:.5f}, loss_target: {:.5f}, homo loss: {:.5f} '\
                        .format(i, loss_inner, loss_target, loss_homo))
                print("acc_train_clean: {:.4f}, ASR_train_attach: {:.4f}, ASR_train_outter: {:.4f}"\
                        .format(acc_train_clean,acc_train_attach,acc_train_outter))
        if args.debug:
            print("load best weight based on the loss outter")
        self.trojan.load_state_dict(self.weights)
        self.trojan.eval()

        # torch.cuda.empty_cache()
    def get_all_trojan_edge(self):
        return deepcopy(self.all_inject_edges)
    
    def get_poisoned(self):

        with torch.no_grad():
            poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger(self.idx_attach,self.features,self.edge_index,self.edge_weights,self.device)
        poison_labels = self.labels
        poison_edge_index = poison_edge_index[:,poison_edge_weights>0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights>0.0]
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels

# %%
def tensor2sparse_coo_tensor(edge_index, n, device):
    num_edges = edge_index.size(1)
    indices = torch.stack([edge_index[0], edge_index[1]], dim=0)  # 堆叠源节点和目标节点索引
    values = torch.ones(num_edges, dtype=torch.float).to(device)  # 边的值，这里假设为1
    # 现在，我们可以创建稀疏张量
    # 注意：torch.sparse_coo_tensor 需要一个大小为 (max_num_nodes, max_num_nodes) 的密集形状
    sparse_tensor = torch.sparse_coo_tensor(indices, values, (n, n))
    return sparse_tensor
