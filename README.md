## 1. Program Overview
* `./run_DTGBA_under_EdgePruning.py`: The program to run DTGBA under edge pruning.
* `./run_DTGBA_under_RandomEdgeDropping.py`: The program to run DTGBA under random edge dropping.

## 2. Requirements
Please see requirements.txt.

## 3. Datasets
In our experiments, we use three datasets, i.e., Cora, Pubmed, and OGB-Arxiv. You need to create a folder called "data" under DTGBA-main. Provide a dataset that will be downloaded automatically when you run the corresponding code. For Cora and Pubmed, you'll use torch_geometric.datasets's Planetoid. For ogbn-arxiv, you will use ogb.nodeproppred's PygNodePropPredDataset

## 4. Our Proposed Method: DTGBA
Graph Neural Networks (GNNs) exhibit significant vulnerability when facing graph backdoor attacks. Specifically, graph backdoor attacks inject triggers and target labels into poisoned nodes during the training phase to create a backdoored GNN. During the testing phase, triggers are added to target nodes, causing them to be misclassified as the target class. However, existing graph backdoor attacks lack sufficient imperceptibility and can be easily resisted by random edge dropping-based defense, limiting their effectiveness. To address these issues, we propose Dual Trigger Graph Backdoor Attack (DTGBA). First, we deploy an imperceptible injected trigger generator and multiple discriminators, driving the imperceptibility of the injected triggers through adversarial game between them. Additionally, we introduce a feature mask learner to extract the high-impact and low-impact feature dimensions of nodes, and then create a feature-based trigger by modifying the key feature dimensions of poisoned/target nodes. This results in a dual-trigger mode, combining imperceptible injected triggers and feature triggers, ensuring that the backdoor implantation can still be effective after the injected triggers are removed by random edge dropping. Finally, we conduct extensive experiments to demonstrate that our proposed method achieves superior performance.

## 5. Reproduce Instructions
The instructions of edge pruning are as follows (The surrogate model is set to 2-layer GCN):
```
python run_DTGBA_under_EdgePruning.py --model GCN --dataset Cora --defense_mode prune --prune_thr 0.5

python run_DTGBA_under_EdgePruning.py --model GCN --dataset Pubmed --defense_mode prune --prune_thr 0.5

python run_DTGBA_under_EdgePruning.py --model GCN --dataset ogbn-arxiv --defense_mode prune --prune_thr 0.7
```

And the instructions of RIGBD are as follows (The surrogate model is set to 2-layer GCN.):
```
python run_DTGBA_under_RandomEdgeDropping.py --model GCN --dataset Cora --defense_mode edgedropping --KK 10 --dropping_rate 0.5

python run_DTGBA_under_RandomEdgeDropping.py --model GCN --dataset Pubmed --defense_mode edgedropping --KK 10 --dropping_rate 0.5

python run_DTGBA_under_RandomEdgeDropping.py --model GCN --dataset ogbn-arxiv --defense_mode edgedropping --KK 10 --dropping_rate 0.5
```

Other hyperparameter's settings have been given in parser.add_argument of the code.

## 6. Code of Baselines
#### SBA-Samp
From Zhang, Zaixi, et al. "Backdoor Attacks to Graph Neural Networks" [[paper](https://arxiv.org/abs/2006.11165), [code](https://github.com/zaixizhang/graphbackdoor)].

#### SBA-Gen
This is a variant of SBA-Samp, which uses generated features for trigger nodes. Features are from a Gaussian distribution whose mean and variance is computed from real nodes.

#### GTA
From Xi, Zhaohan, et al. "Graph Backdoor" [[paper](https://arxiv.org/abs/2006.11890), [code](https://github.com/HarrialX/GraphBackdoor)].

#### UGBA
From Dai, Enyan, et al. "Unnoticeable backdoor attacks on graph neural networks" [[paper](https://dl.acm.org/doi/10.1145/3543507.3583392), [code](https://github.com/ventr1c/UGBA)].

#### DPGBA
From Zhang, Zhiwei, et al. "Rethinking Graph Backdoor Attacks: A Distribution-Preserving Perspective" [[paper](https://dl.acm.org/doi/10.1145/3637528.3671910), [code](https://github.com/zzwjames/DPGBA)].
