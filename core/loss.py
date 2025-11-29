from typing import List

import torch
import torch.nn as nn

from utils.utility import euclidean_distance, manhattan_distance


def weights_flatten(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def model_distance(x: torch.Tensor, y: List[torch.Tensor] = None, type='euclidean') -> torch.Tensor:
    x = x.view(1, -1)
    y = torch.stack(y, dim=0)
    if type == 'euclidean':
        return torch.sum(euclidean_distance(x, y))
    elif type == 'manhattan':
        return torch.sum(manhattan_distance(x, y))


def inter_cluster_loss(model_weight, siblings_weights):
    return model_distance(weights_flatten(model_weight), siblings_weights) / len(siblings_weights)


def intra_cluster_loss(model_weight, predecessor):
    return model_distance(weights_flatten(model_weight), [predecessor])


class SOFA_FL_Loss(nn.Module):
    def __init__(self, local_objective_func, alpha=0.5, beta=0.5):
        super().__init__()
        self.local_objective_func = local_objective_func
        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs, targets, model, siblings, predecessor):
        local_loss = self.local_objective_func(outputs, targets)
        inter_loss = self.alpha * inter_cluster_loss(model, siblings) if siblings is not None else 0
        intra_loss = self.beta * intra_cluster_loss(model, predecessor) if predecessor is not None else 0

        return local_loss + inter_loss + intra_loss
