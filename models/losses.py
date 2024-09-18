"""
Refers to https://github.com/houliangict/SlimGAN
Loss functions definitions.
"""
import torch
import torch.nn.functional as F


def _bce_loss_with_logits(output, labels, **kwargs):
    r"""
    Wrapper for BCE loss with logits.
    """
    # print('output',output)
    return F.binary_cross_entropy_with_logits(output, labels, **kwargs)


def _loss_with_logits(output, labels, **kwargs):
    r"""
    Wrapper for BCE loss with logits.
    """
    return F.cross_entropy(output, labels, **kwargs)


def minimax_loss_gen(output_fake, real_label_val=1.0, **kwargs):
    r"""
    Standard minimax loss for GANs through the BCE Loss with logits fn.
    Args:
        output (Tensor): Discriminator output logits.
        labels (Tensor): Labels for computing cross entropy.
    Returns:
        Tensor: A scalar tensor loss output.      
    """
    real_labels = torch.full((output_fake.shape[0],1),
                            real_label_val,
                            device=output_fake.device)
    loss = _bce_loss_with_logits(output_fake, real_labels, **kwargs)

    return loss


def minimax_loss_dis(output_fake,
                     output_real,
                     real_label_val=1.0,
                     fake_label_val=0.0,
                     **kwargs):
    r"""
    Standard minimax loss for GANs through the BCE Loss with logits fn.
    Args:
        output_fake (Tensor): Discriminator output logits for fake images.    
        output_real (Tensor): Discriminator output logits for real images.
        real_label_val (int): Label for real images.
        fake_label_val (int): Label for fake images.
        device (torch.device): Torch device object for sending created data.
    Returns:
        Tensor: A scalar tensor loss output.      
    """
    # Produce real and fake labels.

    fake_labels = torch.full((output_fake.shape[0],1),
                            fake_label_val,
                            device=output_fake.device)
    real_labels = torch.full((output_real.shape[0],1),
                            real_label_val,
                            device=output_real.device)

    # FF, compute loss and backprop D
    errD_fake = _bce_loss_with_logits(output=output_fake,
                                      labels=fake_labels,
                                      **kwargs)

    errD_real = _bce_loss_with_logits(output=output_real,
                                      labels=real_labels,
                                      **kwargs)

    # Compute cumulative error
    loss = errD_real + errD_fake

    return loss


def ns_loss_gen(output_fake):
    r"""
    Non-saturating loss for generator.
    Args:
        output_fake (Tensor): Discriminator output logits for fake images.
    Returns:
        Tensor: A scalar tensor loss output.    
    """
    output_fake = torch.sigmoid(output_fake)

    return -torch.mean(torch.log(output_fake + 1e-8))


def wasserstein_loss_dis(output_real, output_fake):
    r"""
    Computes the wasserstein loss for the discriminator.
    Args:
        output_real (Tensor): Discriminator output logits for real images.
        output_fake (Tensor): Discriminator output logits for fake images.
    Returns:
        Tensor: A scalar tensor loss output.        
    """
    loss = -1.0 * output_real.mean() + output_fake.mean()

    return loss


def wasserstein_loss_gen(output_fake):
    r"""
    Computes the wasserstein loss for generator.
    Args:
        output_fake (Tensor): Discriminator output logits for fake images.
    Returns:
        Tensor: A scalar tensor loss output.
    """
    loss = -output_fake.mean()

    return loss


def hinge_loss_dis(output_fake, output_real):
    r"""
    Hinge loss for discriminator.
    Args:
        output_fake (Tensor): Discriminator output logits for fake images.
        output_real (Tensor): Discriminator output logits for real images.
    Returns:
        Tensor: A scalar tensor loss output.        
    """
    loss = F.relu(1.0 - output_real).mean() + \
           F.relu(1.0 + output_fake).mean()

    return loss


def hinge_loss_gen(output_fake):
    r"""
    Hinge loss for generator.
    Args:
        output_fake (Tensor): Discriminator output logits for fake images.
    Returns:
        Tensor: A scalar tensor loss output.      
    """
    loss = -output_fake.mean()

    return loss


def compute_gan_loss(loss_type, output):
    r"""
    Computes GAN loss for generator.
    Args:
        output (Tensor): A batch of output logits from the discriminator of shape (N, 1).
    Returns:
        Tensor: A batch of GAN losses for the generator.
    """
    # Compute loss and backprop
    if loss_type == "gan":
        lossG = minimax_loss_gen(output)

    elif loss_type == "ns":
        lossG = ns_loss_gen(output)

    elif loss_type == "hinge":
        lossG = hinge_loss_gen(output)

    elif loss_type == "wasserstein":
        lossG = wasserstein_loss_gen(output)

    else:
        raise ValueError("Invalid loss_type {} selected.".format(loss_type))

    return lossG


def pdist(a, b, p=2,eps=1e-16):
    return ((a-b).abs().pow(p).sum(-1) + eps).pow(1/p)

def percept_loss(all_emb_list, inj_idx):
    n_inj = inj_idx.shape[0]
    graphps_mx = torch.zeros(n_inj, n_inj, device=all_emb_list[0].device)
    for layer in range(len(all_emb_list)):
        # print('layer',layer)
        emb = all_emb_list[layer][inj_idx]
        norm_emb = F.normalize(emb)
        layer_loss = pdist(norm_emb.unsqueeze(1), norm_emb.unsqueeze(0))
        # print('layer_loss',layer_loss.max(),layer_loss.min())
        graphps_mx += layer_loss
    graphps_triu = torch.triu(graphps_mx)#上三角部分，对角线和下三角部分保持为0
    indices = graphps_triu.nonzero().t()
    graphps = graphps_triu[indices[0],indices[1]].sum() / indices.shape[1]
    return -graphps

def tensor2sparse_coo_tensor(edge_index, n, device):
    num_edges = edge_index.size(1)
    indices = torch.stack([edge_index[0], edge_index[1]], dim=0)  # 堆叠源节点和目标节点索引
    values = torch.ones(num_edges, dtype=torch.float).to(device)  # 边的值，这里假设为1
    # 现在，我们可以创建稀疏张量
    # 注意：torch.sparse_coo_tensor 需要一个大小为 (max_num_nodes, max_num_nodes) 的密集形状
    sparse_tensor = torch.sparse_coo_tensor(indices, values, (n, n))
    return sparse_tensor

def add_feature(f1, f2):
    pass

def add_edge_index(e1, e2):
    pass

def compute_D_loss(real_batch, fake_batch, netD, adj_tensor, feat, new_feat, new_adj_tensor,  n , new_n, device,clipfeat=False):
    real_batch_size = len(real_batch)
    fake_batch_size = len(fake_batch)

    new_feat = torch.cat((feat, new_feat), dim=0)
    new_adj_tensor = torch.cat((adj_tensor, new_adj_tensor), dim=1)

    adj_tensor = tensor2sparse_coo_tensor(adj_tensor, n, device)
    new_adj_tensor = tensor2sparse_coo_tensor(new_adj_tensor, new_n, device)

    real_rate = real_batch_size/(real_batch_size+fake_batch_size)
    fake_rate = fake_batch_size/(real_batch_size+fake_batch_size)
    pred_real = netD(feat, adj_tensor)[1]

    if clipfeat:
        clip_feat = torch.clamp(new_feat.detach(),feat.min(),feat.max())
        pred_fake_D = netD(clip_feat,new_adj_tensor.detach())[1]
    else:
        pred_fake_D = netD(new_feat.detach(),new_adj_tensor.detach())[1]
    loss_D = netD.compute_gan_loss(pred_real[real_batch], pred_fake_D[fake_batch])

    real_label = torch.full((real_batch_size,1), 1.0, device=pred_real.device)
    fake_label = torch.full((fake_batch_size,1), 0.0, device=pred_real.device)
    acc_real = netD.compute_acc(pred_real[real_batch], real_label) 
    acc_fake = netD.compute_acc(pred_fake_D[fake_batch], fake_label) 
    acc_D =  acc_real * real_rate + acc_fake * fake_rate
    return loss_D, acc_D, acc_real, acc_fake
