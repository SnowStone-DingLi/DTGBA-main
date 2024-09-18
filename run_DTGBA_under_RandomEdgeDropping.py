import imp
import time
import argparse
import numpy as np
import torch
from copy import deepcopy
from torch_geometric.datasets import Planetoid, Reddit2, Flickr, CitationFull, Amazon, Twitch, Actor

# from torch_geometric.loader import DataLoader
from help_funcs import prune_unrelated_edge, prune_unrelated_edge_isolated, get_rand_edge_dropping
import scipy.sparse as sp
from torch.nn.functional import kl_div

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
                    default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN', 'GAT', 'GraphSage', 'GIN'])
parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                    help='Dataset',
                    choices=['Cora', 'Pubmed', 'ogbn-arxiv'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=0.5)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int, default=400, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int, default=1, help='Number of inner')
# backdoor setting
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--trigger_size', type=int, default=3,
                    help='trigger_size')
parser.add_argument('--use_vs_number', action='store_true', default=True,
                    help="if use detailed number to decide Vs")
parser.add_argument('--vs_ratio', type=float, default=0,
                    help="ratio of poisoning nodes relative to the full graph")
parser.add_argument('--vs_number', type=int, default=565,
                    help="number of poisoning nodes relative to the full graph")
# defense setting
parser.add_argument('--defense_mode', type=str, default="edgedropping",
                    choices=['prune', 'isolate', 'none','edgedropping'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.0,
                    help="Threshold of prunning edges")
parser.add_argument('--target_loss_weight', type=float, default=1,
                    help="Weight of optimize outter trigger generator")
parser.add_argument('--homo_loss_weight', type=float, default=100,
                    help="Weight of optimize similarity loss")
parser.add_argument('--homo_boost_thrd', type=float, default=0.5,
                    help="Threshold of increase similarity")
# attack setting
parser.add_argument('--dis_weight', type=float, default=1,
                    help="Weight of cluster distance")
parser.add_argument('--selection_method', type=str, default='cluster_degree',
                    choices=['loss', 'conf', 'cluster', 'none', 'cluster_degree'],
                    help='Method to select idx_attach for training trojan model (none means randomly select)')
parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN', 'GAT', 'GraphSage', 'GIN'],
                    help='Model used to attack')
parser.add_argument('--evaluate_mode', type=str, default='overall',
                    choices=['overall', '1by1'],
                    help='Model used to attack')
# GPU setting
parser.add_argument('--device_id', type=int, default=0,
                    help="id of device")
# GAN setting
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--st_epoch', type=int, default=0, help='number of epochs')
parser.add_argument('--lr_D', default=1e-3, type=float, help='learning rate of Discriminator')
parser.add_argument('--Dopt', type=int, default=20, help='Discriminator optimize Dopt times, G optimize 1 time.')
parser.add_argument('--D_sn', type=str, default='none', choices=['none', 'SN'],
                    help='whether Discriminator use spectral_norm')
parser.add_argument('--D_type', type=str, default='gin', help='Discriminator type')
parser.add_argument('--loss_type', type=str, default='gan', choices=['gan', 'ns', 'hinge', 'wasserstein'],
                    help='Loss type.')
parser.add_argument('--step_size', type=int, default=100, help='Step size of the optimizer scheduler')
parser.add_argument('--patience', type=int, default=500, help='Patience of feature optimization')

parser.add_argument('--atk_alpha', type=float, default=1, help='the coefficient of GAN loss in G loss')
parser.add_argument('--alpha', type=float, default=1, help='the coefficient of GAN loss in G loss')
parser.add_argument('--beta', type=float, default=0.01, help='the coefficient of diversity loss in G loss')
parser.add_argument('--clipfeat', type=bool, default=False,
                    help='the coefficient of GAN when choosing edges to connect')
parser.add_argument('--wd_D', type=float, default=0.1)
parser.add_argument('--K', type=int, default=3, help='Number of Discriminators')
# feature trigger settings
parser.add_argument('--topk_ratio', type=float, default=0.01, help='Number of Discriminators')
parser.add_argument('--feature_type', type=bool, default=True, help='True: discrete, False: continuous')
# Rand Edge Dropping
parser.add_argument('--KK', type=int, default=10,
                    help="Number of Rand Edge Dropping")
parser.add_argument('--dropping_rate', type=float, default=0.5,
                    help="probability of Edge Dropping")

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(args)
# %%
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T

transform = T.Compose([T.NormalizeFeatures()])

if (args.dataset == 'Cora' or args.dataset == 'Citeseer' or args.dataset == 'Pubmed'):
    dataset = Planetoid(root='./data/', \
                        name=args.dataset, \
                        transform=transform)
elif (args.dataset == 'Flickr'):
    dataset = Flickr(root='./data/Flickr/', \
                     transform=transform)
elif (args.dataset == 'ogbn-arxiv'):
    from ogb.nodeproppred import PygNodePropPredDataset
    # Download and process data at './dataset/ogbg_molhiv/'
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data/')
    split_idx = dataset.get_idx_split()
elif (args.dataset == 'Cora_ML'):
    dataset = CitationFull(root='./data/', \
                        name='cora_ml', \
                        transform=transform)
    # split_idx = dataset.get_idx_split()
elif (args.dataset == 'Computers' or args.dataset == 'Photo'):
    dataset = Amazon(root='./data/', \
                        name=args.dataset, \
                        transform=transform)
elif (args.dataset in ['DE', 'EN','ES', 'FR', 'PT' , 'RU']):
    dataset = Twitch(root='./data/', \
                        name=args.dataset, \
                        transform=transform)
elif (args.dataset == 'Actor'):
    dataset = Actor(root='./data/', \
                        transform=transform)
data = dataset[0].to(device)

if (args.dataset == 'ogbn-arxiv'):
    nNode = data.x.shape[0]
    setattr(data, 'train_mask', torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)
elif (args.dataset == 'Cora_ML'):
    nNode = data.x.shape[0]
    setattr(data, 'train_mask', torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    # data.y = data.y.squeeze(1)
elif (args.dataset == 'Computers' or args.dataset == 'Photo' or args.dataset == 'Actor'):
    nNode = data.x.shape[0]
    setattr(data, 'train_mask', torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
elif (args.dataset in ['DE', 'EN','ES', 'FR', 'PT' , 'RU']):
    nNode = data.x.shape[0]
    setattr(data, 'train_mask', torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)

# we build our own train test split
# %%
from utils import get_split

data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args, data, device)

from torch_geometric.utils import to_undirected
from utils import subgraph

data.edge_index = to_undirected(data.edge_index)
train_edge_index, _, edge_mask = subgraph(torch.bitwise_not(data.test_mask), data.edge_index, relabel_nodes=False)
mask_edge_index = data.edge_index[:, torch.bitwise_not(edge_mask)]  # 未被子图选中的边

# In[9]:

from sklearn_extra import cluster
from models.DTGBA import Backdoor
from models.construct import model_construct
import heuristic_selection as hs

# from kmeans_pytorch import kmeans, kmeans_predict

# filter out the unlabeled nodes except from training nodes and testing nodes, nonzero() is to get index, flatten is to get 1-d tensor
unlabeled_idx = (torch.bitwise_not(data.test_mask) & torch.bitwise_not(data.train_mask)).nonzero().flatten()
if (args.use_vs_number):
    size = args.vs_number
else:
    size = int((len(data.test_mask) - data.test_mask.sum()) * args.vs_ratio)
print("#Attach Nodes:{}".format(size))
assert size > 0, 'The number of selected trigger nodes must be larger than 0!'

# here is randomly select poison nodes from unlabeled nodes
if (args.selection_method == 'none'):
    idx_attach = hs.obtain_attach_nodes(args, unlabeled_idx, size)
elif (args.selection_method == 'cluster'):
    idx_attach = hs.cluster_distance_selection(args, data, idx_train, idx_val, idx_clean_test, unlabeled_idx,
                                               train_edge_index, size, device)
    idx_attach = torch.LongTensor(idx_attach).to(device)
elif (args.selection_method == 'cluster_degree'):
    if (args.dataset == 'Pubmed'):
        idx_attach = hs.cluster_degree_selection_seperate_fixed(args, data, idx_train, idx_val, idx_clean_test,
                                                                unlabeled_idx, train_edge_index, size, device)
    else:
        idx_attach = hs.cluster_degree_selection(args, data, idx_train, idx_val, idx_clean_test, unlabeled_idx,
                                                 train_edge_index, size, device)
    idx_attach = torch.LongTensor(idx_attach).to(device)
print("idx_attach: {}".format(idx_attach))  # 所要附着的节点idx
unlabeled_idx = torch.tensor(list(set(unlabeled_idx.cpu().numpy()) - set(idx_attach.cpu().numpy()))).to(device)
print(unlabeled_idx)
# In[10]:
from models.GCN2 import GCN2
from models.GIN import GIN
from models.discriminator import Discriminator
from torch_geometric.explain import GNNExplainer
from models.FeatureMaskLearner import FeatureMaskLearner

features = data.x
labels = data.y

edge_weights = torch.ones([data.edge_index.shape[1]]).to(device)
test_model = model_construct(args, args.model, data, device).to(device)
test_model.fit(data.x, data.edge_index, edge_weights, data.y, idx_train, idx_val, train_iters=args.epochs, verbose=False)
output = test_model(data.x, data.edge_index, edge_weights)
# print(output)
ca = test_model.test(data.x, data.edge_index, edge_weights, data.y, idx_clean_test)
# print(ca)
print(output.argmax(dim=1))
pred_labels = output.argmax(dim=1)
print(pred_labels.shape)
logp = test_model(data.x, data.edge_index)

pred_labels = logp.max(1)[1]
pred_labels_np = pred_labels.cpu().numpy()
all_n = len(pred_labels)

real =  pred_labels == data.y
print(all_n)
print("clean acc: ", real.sum().item()*1.0/ all_n)

explainer = FeatureMaskLearner(
    model=test_model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)

indices = torch.where(pred_labels == args.target_class)[0]
print(indices)
print(len(indices))

explanation = explainer(data.x, data.edge_index, index=indices)
node_mask = explanation.get('node_mask')
print(node_mask.shape)
feat_labels = range(node_mask.size(1))
score = node_mask.sum(dim=0)
top_k = int(args.topk_ratio * data.x.shape[1])

score = score.cpu().numpy()
# most important
top_k_index = np.argsort(score)[::-1][0:top_k]
print("top_k_index: ", top_k_index)
print(data.x.shape)
select_feature = data.x[indices]
top_k_index_tensor = torch.tensor(top_k_index.copy())
select_feature = select_feature[:,top_k_index_tensor]
print(select_feature.shape)
print(len(indices))

sum_select_feature = select_feature.sum(dim = 0) / len(indices)
print(sum_select_feature.shape)
print(sum_select_feature)
# 
print('================')
# least important
least_k_index = np.argsort(score)[0:top_k]
print("least_k_index: ", least_k_index)
print(data.x.shape)
select_feature = data.x[indices]
least_k_index_tensor = torch.tensor(least_k_index.copy())
select_feature = select_feature[:,least_k_index_tensor]
print(select_feature.shape)
print(len(indices))

sum_select_feature = select_feature.sum(dim = 0) / len(indices)
print(sum_select_feature.shape)
print(sum_select_feature)

print("=================")
vector_pos = torch.zeros(len(idx_attach), data.x.shape[1])
vector_pos[: , top_k_index_tensor] = 1.0

vector_neg = torch.ones(len(idx_attach), data.x.shape[1])
vector_neg[: , least_k_index_tensor] = 0.0

vector_pos = vector_pos.to(device)
print(vector_pos)
print(vector_pos.shape)

vector_neg = vector_neg.to(device)
print(vector_neg)
print(vector_neg.shape)

# train trigger generator

rep_net = GIN(2, 2, features.shape[1],
                   64, labels.max().item()+1, 0.5,
                   False, 'sum', 'sum').to(device)

rep_net.eval()

netD = Discriminator(features.shape[1], args.hidden, 1, args.loss_type, 0, args.D_sn, d_type=args.D_type).to(device)
optimizer_D = torch.optim.Adam([{'params': netD.parameters()}], lr=args.lr_D, weight_decay=args.wd_D)

model = Backdoor(args, device)
model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_attach, unlabeled_idx, netD, optimizer_D, args.Dopt, rep_net ,'gan', vector_neg, vector_pos, top_k_index_tensor, least_k_index_tensor)
poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned()

if (args.defense_mode == 'prune'):
    poison_edge_index, poison_edge_weights = prune_unrelated_edge(args, poison_edge_index, poison_edge_weights,
                                                                  poison_x, device, large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train, idx_attach]).to(device)
elif (args.defense_mode == 'isolate'):
    poison_edge_index, poison_edge_weights, rel_nodes = prune_unrelated_edge_isolated(args, poison_edge_index,
                                                                                      poison_edge_weights, poison_x,
                                                                                      device, large_graph=False)
    bkd_tn_nodes = torch.cat([idx_train, idx_attach]).tolist()
    bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
elif(args.defense_mode =='edge_droppoing'):
    multi_graph_edge_index = []
    multi_graph_edge_weight = []
    for i in range(args.KK):
        a , b = get_rand_edge_dropping(args,poison_edge_index,poison_edge_weights,poison_x,device)
        multi_graph_edge_index.append(a)
        multi_graph_edge_weight.append(b)
        bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device)
else:
    bkd_tn_nodes = torch.cat([idx_train, idx_attach]).to(device)
print("precent of left attach nodes: {:.3f}".format(
    len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist())) / len(idx_attach)))


models = ['GCN']
total_overall_asr = 0
total_overall_ca = 0
for test_model in models:
    args.test_model = test_model
    rs = np.random.RandomState(args.seed)
    seeds = rs.randint(1000, size=5)
    seeds = [11,12,13,14,15]
    overall_asr = 0
    overall_ca = 0
    for seed in seeds:
        args.seed = seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        print(args)
        # %%
        test_model = model_construct(args, args.test_model, data, device).to(device)
        test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,
                       train_iters=args.epochs, verbose=False)

        output = test_model(poison_x, poison_edge_index, poison_edge_weights)
        train_attach_rate = (output.argmax(dim=1)[idx_attach] == args.target_class).float().mean()
        print("target class rate on Vs: {:.4f}".format(train_attach_rate))
        # %%
        induct_edge_index = torch.cat([poison_edge_index, mask_edge_index], dim=1)
        induct_edge_weights = torch.cat(
            [poison_edge_weights, torch.ones([mask_edge_index.shape[1]], dtype=torch.float, device=device)])
        clean_acc = test_model.test(poison_x, induct_edge_index, induct_edge_weights, data.y, idx_clean_test)

        print("accuracy on clean test nodes: {:.4f}".format(clean_acc))
        if (args.evaluate_mode == '1by1'):
            from torch_geometric.utils import k_hop_subgraph

            overall_induct_edge_index, overall_induct_edge_weights = induct_edge_index.clone(), induct_edge_weights.clone()
            asr = 0
            flip_asr = 0
            flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
            for i, idx in enumerate(idx_atk):
                idx = int(idx)
                sub_induct_nodeset, sub_induct_edge_index, sub_mapping, sub_edge_mask = k_hop_subgraph(node_idx=[idx],
                                                                                                       num_hops=2,
                                                                                                       edge_index=overall_induct_edge_index,
                                                                                                       relabel_nodes=True)
                ori_node_idx = sub_induct_nodeset[sub_mapping]
                relabeled_node_idx = sub_mapping
                sub_induct_edge_weights = torch.ones([sub_induct_edge_index.shape[1]]).to(device)
                with torch.no_grad():
                    # inject trigger on attack test nodes (idx_atk)'''
                    induct_x, induct_edge_index, induct_edge_weights = model.inject_trigger(relabeled_node_idx,
                                                                                            poison_x[
                                                                                                sub_induct_nodeset],
                                                                                            sub_induct_edge_index,
                                                                                            sub_induct_edge_weights,
                                                                                            device)
                    induct_x, induct_edge_index, induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(), induct_edge_weights.clone().detach()
                    # # do pruning in test datas'''
                    if (args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
                        induct_edge_index, induct_edge_weights = prune_unrelated_edge(args, induct_edge_index,
                                                                                      induct_edge_weights, induct_x,
                                                                                      device, False)
                    # attack evaluation
                    output = test_model(induct_x, induct_edge_index, induct_edge_weights)
                    train_attach_rate = (output.argmax(dim=1)[relabeled_node_idx] == args.target_class).float().mean()
                    asr += train_attach_rate
                    if (data.y[idx] != args.target_class):
                        flip_asr += train_attach_rate
                    induct_x, induct_edge_index, induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(), induct_edge_weights.cpu()
                    output = output.cpu()
                    final_x = induct_x
                    final_edge_index = induct_edge_index
            asr = asr / (idx_atk.shape[0])
            flip_asr = flip_asr / (flip_idx_atk.shape[0])
            print("Idx_atk:")
            print(idx_atk)
            print("Overall ASR: {:.4f}".format(asr))
            print("Flip ASR: {:.4f}/{} nodes".format(flip_asr, flip_idx_atk.shape[0]))
        elif (args.evaluate_mode == 'overall'):
            # %% inject trigger on attack test nodes (idx_atk)'''
            induct_x, induct_edge_index, induct_edge_weights = model.inject_trigger(idx_atk, poison_x,
                                                                                    induct_edge_index,
                                                                                    induct_edge_weights, device)
            
            induct_x, induct_edge_index, induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(), induct_edge_weights.clone().detach()
            # do pruning in test datas'''
            if (args.defense_mode == 'prune' or args.defense_mode == 'isolate'):
                induct_edge_index, induct_edge_weights = prune_unrelated_edge(args, induct_edge_index,
                                                                              induct_edge_weights, induct_x, device) 
            elif(args.defense_mode == 'edgedropping'):
                multi_graph_edge_index = []
                multi_graph_edge_weight = []
                for i in range(args.KK):
                    a , b = get_rand_edge_dropping(args,induct_edge_index,induct_edge_weights,poison_x,device)
                    multi_graph_edge_index.append(a)
                    multi_graph_edge_weight.append(b)  
            # attack evaluation
            output = test_model(induct_x, induct_edge_index, induct_edge_weights)
            output_with_log_softmax = output
            output_with_log_softmax_array = []

            for i in range(args.KK):
                output_with_log_softmax_array.append(test_model(induct_x,multi_graph_edge_index[i],multi_graph_edge_weight[i]))
            s=[]

            print(output_with_log_softmax.shape)
            for i in range(args.KK):
                s.append(kl_div( output_with_log_softmax, output_with_log_softmax_array[i], reduction='none', log_target=True))
                s[i] = torch.sum(s[i], dim=1)
            final_s=s[0]

            for i in range(args.KK):
                if i==0:
                    print('do nothing')
                else:
                    final_s = final_s +s[i]

            print(final_s)
            print(final_s.shape)

            final_s = final_s[bkd_tn_nodes]
            indices = final_s.argsort(descending=True)

            identified_target_class = poison_labels[bkd_tn_nodes[ indices[0]]]
            print("identified_target_class: ",identified_target_class)
            identified_poison_nodes = []
            cnt = 0
            for i in range(len(indices)):
                if poison_labels[bkd_tn_nodes[ indices[i]]] == identified_target_class:
                    identified_poison_nodes.append(bkd_tn_nodes[ indices[i]].item())
                    cnt=0
                elif poison_labels[bkd_tn_nodes[ indices[i]]] != identified_target_class and cnt==0:
                    cnt+=1
                elif poison_labels[bkd_tn_nodes[ indices[i]]] != identified_target_class and cnt==1:
                    break
            print('final_identified_poison_nodes:')
            print(identified_poison_nodes)
            print('idx_atk:')
            print(idx_atk)

            list_of_bkd_tn_nodes= bkd_tn_nodes.tolist()
            identified_clean_nodes = torch.tensor(list(set(list_of_bkd_tn_nodes) - set(identified_poison_nodes)))  
            identified_poison_nodes = torch.tensor(identified_poison_nodes)
            # print(len(bkd_tn_nodes))
            # print(len(identified_clean_nodes))
            # print(len(identified_poison_nodes))

            test_model = model_construct(args,'RIGBD',data,device).to(device) 
            test_model.fit(poison_x, poison_edge_index, poison_edge_weights, poison_labels, identified_clean_nodes, identified_poison_nodes,identified_target_class, idx_val,train_iters=args.epochs,verbose=False)

            output = test_model(induct_x,induct_edge_index,induct_edge_weights)

            train_attach_rate = (output.argmax(dim=1)[idx_atk] == args.target_class).float().mean()
            print("ASR: {:.4f}".format(train_attach_rate))
            asr = train_attach_rate
            flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
            flip_asr = (output.argmax(dim=1)[flip_idx_atk] == args.target_class).float().mean()
            print("Flip ASR: {:.4f}/{} nodes".format(flip_asr, flip_idx_atk.shape[0]))
            ca = test_model.test(induct_x, induct_edge_index, induct_edge_weights, data.y, idx_clean_test)
            print("CA: {:.4f}".format(ca))

            induct_x, induct_edge_index, induct_edge_weights = induct_x.cpu(), induct_edge_index.cpu(), induct_edge_weights.cpu()
            output = output.cpu()
            final_x = induct_x
            final_edge_index = induct_edge_index
        overall_asr += asr
        overall_ca += clean_acc

        test_model = test_model.cpu()

    overall_asr = overall_asr / len(seeds)
    overall_ca = overall_ca / len(seeds)
    print("Overall ASR: {:.4f} ({} model, Seed: {})".format(overall_asr, args.test_model, args.seed))
    print("Overall Clean Accuracy: {:.4f}".format(overall_ca))

    total_overall_asr += overall_asr
    total_overall_ca += overall_ca
    test_model.to(torch.device('cpu'))
    torch.cuda.empty_cache()
total_overall_asr = total_overall_asr / len(models)
total_overall_ca = total_overall_ca / len(models)
print("Total Overall ASR: {:.4f} ".format(total_overall_asr))
print("Total Clean Accuracy: {:.4f}".format(total_overall_ca))
print('sim analysis')
print(data.x.shape)
