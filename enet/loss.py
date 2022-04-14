import torch
import torch.nn.functional as tf
import torch.nn as nn


def discriminative_loss_single(embedding, inst_label,
                               delta_v, delta_d,
                               param_var, param_dist, param_reg):
    c, h, w = embedding.size()
    if inst_label.is_cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    num_classes = len(torch.unique(inst_label)) - 1
    h, w = inst_label.shape
    inst_masks = torch.zeros(num_classes, h, w).byte().to(device)

    for idx, label in enumerate(torch.unique(inst_label)):
        if label == 0:
            continue
        else:
            inst_masks[idx - 1] = (inst_label == label)

    embeddings = []

    for i in range(num_classes):
        feature = torch.transpose(torch.masked_select(embedding,
                                                      inst_masks[i, :, :])
                                  .view(c, -1), 0, 1)
        embeddings.append(feature)

    centers = []
    for feature in embeddings:
        center = torch.mean(feature, dim=0, keepdim=True)
        centers.append(center)

    loss_var = torch.Tensor([0.0]).to(device)
    for feature, center in zip(embeddings, centers):
        dis = torch.norm(feature - center, 2, dim=1) - delta_v
        dis = tf.relu(dis)
        loss_var += torch.mean(dis)
    loss_var /= num_classes

    if num_classes == 1:
        return loss_var, loss_var, torch.zeros(1)  # .to(device)

    centers = torch.cat(centers, dim=0)
    a = centers.repeat(1, num_classes).view(-1, c)
    b = centers.repeat(num_classes, 1)
    distance = torch.norm(a - b, 2, dim=1).view(num_classes, num_classes)

    eye = torch.eye(num_classes).to(device)
    pair_distance = torch.masked_select(distance, eye == 0)

    pair_distance = delta_d - pair_distance
    pair_distance = tf.relu(pair_distance)
    loss_dist = torch.mean(pair_distance).view(-1)

    loss_reg = torch.mean(torch.norm(centers, 2, dim=1)).view(-1)

    loss = param_var * loss_var + param_dist * loss_dist + param_reg * loss_reg
    return loss


def discriminative_loss(embedding_batch, label_batch,
                        delta_v=0.5,
                        delta_d=1.5,
                        param_var=1.0,
                        param_dist=1.0,
                        param_reg=0.001):
    loss = 0
    for embedding, inst_lbl in zip(embedding_batch, label_batch):
        _loss = discriminative_loss_single(embedding, inst_lbl,
                                           delta_v, delta_d,
                                           param_var, param_dist, param_reg)

        loss += _loss

    return loss


def enet_loss(logit, embeddings, labels_bin, labels_inst):
    # preds_bin = torch.argmax(logit, dim=1, keepdim=True)
    # preds_bin_expand = preds_bin.view(preds_bin.shape[0]
    #                                   * preds_bin.shape[1]
    #                                   * preds_bin.shape[2]
    #                                   * preds_bin.shape[3])
    labels_bin_expand = labels_bin.view(labels_bin.shape[0]
                                        * labels_bin.shape[1]
                                        * labels_bin.shape[2])
    bin_count = torch.bincount(labels_bin_expand)
    bin_prop = bin_count.float() / torch.sum(bin_count)
    weight_bin = torch.tensor(1) / (bin_prop + 0.2)

    loss_ce = nn.CrossEntropyLoss(weight=weight_bin)
    loss_bin = loss_ce(logit, labels_bin)

    loss_discr = discriminative_loss(embeddings, labels_inst)

    loss = loss_bin + loss_discr
    return loss
