import torch
import numpy as np
import torch.nn as nn
import math

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


@torch.no_grad()
def update_quantized_weight_values(model, perm_size=16, amount=0.5):
    k_pos = math.ceil(0.5 * amount * perm_size)
    k_neg = math.floor(0.5 * amount * perm_size)
    weights = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(nn.BatchNorm1d) or isinstance(
                nn.BatchNorm2d):
            weights.append(layer.weight.clone().detach().to(device))
    weights_flatten = np.zeros(0)
    weights_shape = []
    weights_start_idx = []
    weight_start_idx = 0
    for weight in weights:
        weights_flatten = np.append(weights_flatten, weight.view(-1).clone().detach().cpu())
        weights_shape.append(weight.shape)
        weights_start_idx.append(weight_start_idx)
        weight_start_idx += len(weight.view(-1))
    weights_flatten = torch.Tensor(weights_flatten).to(device)
    if weights_flatten.numel() % perm_size == 0:
        pos_idx_topk = torch.topk(
            weights_flatten.where(weights_flatten > 0, torch.zeros(weights_flatten.shape)).view(-1, perm_size).abs(),
            k=k_pos)
        neg_idx_topk = torch.topk(
            weights_flatten.where(weights_flatten < 0, torch.zeros(weights_flatten.shape)).view(-1, perm_size).abs(),
            k=k_neg)
        pos_weight_values = weights_flatten.where(weights_flatten > 0, torch.zeros(weights_flatten.shape)).view(-1,
                                                            perm_size).gather(dim=1, index=pos_idx_topk).mean(dim=0)
        neg_weight_values = weights_flatten.where(weights_flatten < 0, torch.zeros(weights_flatten.shape)).view(-1,
                                                            perm_size).gather(dim=1, index=neg_idx_topk).mean(dim=0)
    else:
        n_row = weights_flatten.numel() // perm_size
        weights_flatten_dividable = weights_flatten[:n_row * perm_size]
        weights_flatten_rest = weights_flatten[n_row * perm_size:]
        pos_idx_topk_dividable = torch.topk(
            weights_flatten_dividable.where(weights_flatten_dividable > 0,
                                    torch.zeros(weights_flatten_dividable.shape)).view(-1, perm_size).abs(), k=k_pos)
        neg_idx_topk_dividable = torch.topk(
            weights_flatten_dividable.where(weights_flatten_dividable < 0,
                                    torch.zeros(weights_flatten_dividable.shape)).view(-1, perm_size).abs(), k=k_neg)
        pos_idx_topk_rest = torch.topk(
            weights_flatten_rest.where(weights_flatten_rest > 0, torch.zeros(weights_flatten_rest.shape)).abs(), k=k_pos)
        neg_idx_topk_rest = torch.topk(
            weights_flatten_rest.where(weights_flatten_rest < 0, torch.zeros(weights_flatten_rest.shape)).abs(), k=k_neg)
        pos_weight_values_dividable = weights_flatten_dividable.where(weights_flatten_dividable > 0, torch.zeros(
            weights_flatten_dividable.shape)).view(-1, perm_size).gather(dim=1, index=pos_idx_topk_dividable)
        neg_weight_values_dividable = weights_flatten_dividable.where(weights_flatten_dividable < 0, torch.zeros(
            weights_flatten_dividable.shape)).view(-1, perm_size).gather(dim=1, index=neg_idx_topk_dividable)
        pos_weight_values_rest = weights_flatten_rest.where(weights_flatten_rest > 0, torch.zeros(
            weights_flatten_rest.shape)).gather(dim=1, index=pos_idx_topk_rest)
        neg_weight_values_rest = weights_flatten_rest.where(weights_flatten_rest < 0, torch.zeros(
            weights_flatten_rest.shape)).gather(dim=1, index=neg_idx_topk_rest)
        pos_weight_values = torch.cat([pos_weight_values_dividable, pos_weight_values_rest], dim=0).mean(dim=0)
        neg_weight_values = torch.cat([pos_weight_values_dividable, pos_weight_values_rest], dim=0).mean(dim=0)

    quantized_weight_values = torch.cat([pos_weight_values, neg_weight_values], dim=0)
    for layer in model.modules():
        layer.set_quantized_weight_values(quantized_weight_values)


@torch.no_grad()
def update_masks(model, amount=0.5):
    weights = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(nn.BatchNorm1d) or isinstance(nn.BatchNorm2d):
            weights.append(layer.weight.clone().detach().to(device))
    weights_abs_flatten = np.zeros(0)
    weights_shape = []
    weights_start_idx = []
    weight_start_idx = 0
    for weight in weights:
        weights_abs_flatten = np.append(weights_abs_flatten, weight.abs().view(-1).clone().detach().cpu())
        weights_shape.append(weight.shape)
        weights_start_idx.append(weight_start_idx)
        weight_start_idx += len(weight.view(-1))
    weights_abs_flatten = torch.Tensor(weights_abs_flatten).to(device)
    k = int(len(weights_abs_flatten) * (1 - amount))
    idx_topk = torch.topk(weights_abs_flatten, k=k)[1]
    masks_flatten = torch.zeros(weights_abs_flatten.shape).to(device)
    masks_flatten[idx_topk] = 1
    masks = []
    for i, weight in enumerate(weights):
        if i < (len(weights) - 1):
            mask = masks_flatten[weights_start_idx[i]:weights_start_idx[i+1]].reshape(weights_shape[i])
        else:
            mask = masks_flatten[weights_start_idx[i]:].reshape(weights_shape[i])
        masks.append(mask)
    mask_idx = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(nn.BatchNorm1d) or isinstance(nn.BatchNorm2d):
            layer.set_mask(masks[mask_idx])
            mask_idx += 1


@torch.no_grad()
def prune_weight_abs(param, amount=0.9):
    thr = (len(param.view(-1)) - 1) * amount
    param.view(-1)[torch.argsort(param.abs().view(-1)) < thr] = 0


@torch.no_grad()
def prune_weights_abs(params, amount=0.9):
    params = list(params)
    params_abs_flatten = np.zeros(0)
    params_shape = []
    params_start_idx = []
    param_start_idx = 0
    for param in params:
        params_abs_flatten = np.append(params_abs_flatten, param.abs().view(-1).clone().detach().cpu())
        params_shape.append(param.shape)
        params_start_idx.append(param_start_idx)
        param_start_idx += len(param.view(-1))
    params_abs_flatten = torch.Tensor(params_abs_flatten).to(device)
    k = int(len(params_abs_flatten) * (1-amount))
    idx_topk = torch.topk(params_abs_flatten, k=k)[1]
    masks_flatten = torch.zeros(params_abs_flatten.shape).to(device)
    masks_flatten[idx_topk] = 1
    for i, param in enumerate(params):
        if i < (len(params)-1):
            mask = masks_flatten[params_start_idx[i]:params_start_idx[i+1]].reshape(params_shape[i])
        else:
            mask = masks_flatten[params_start_idx[i]:].reshape(params_shape[i])
        param.mul_(mask)
