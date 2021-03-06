import torch
import numpy as np
import torch.nn as nn
import math

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


@torch.no_grad()
def update_quantized_weight_values(model, group_size=16, num_values=0.5):
    k_pos = math.ceil(0.5 * num_values)
    k_neg = math.floor(0.5 * num_values)
    weights = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.BatchNorm2d):
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
    if weights_flatten.numel() % group_size == 0:
        pos_idx_topk = torch.topk(
            weights_flatten.where(weights_flatten > 0, torch.zeros(weights_flatten.shape)).view(-1, group_size).abs(),
            k=k_pos)[1]
        neg_idx_topk = torch.topk(
            weights_flatten.where(weights_flatten < 0, torch.zeros(weights_flatten.shape)).view(-1, group_size).abs(),
            k=k_neg)[1]
        pos_weight_values = weights_flatten.where(weights_flatten > 0, torch.zeros(weights_flatten.shape)).view(-1,
                                                                                                                group_size).gather(dim=1, index=pos_idx_topk).mean(dim=0)
        neg_weight_values = weights_flatten.where(weights_flatten < 0, torch.zeros(weights_flatten.shape)).view(-1,
                                                                                                                group_size).gather(dim=1, index=neg_idx_topk).mean(dim=0)
    else:
        n_row = math.ceil(weights_flatten.numel() / group_size)
        extended_weights_flatten = torch.zeros(n_row * group_size)
        extended_weights_flatten[:weights_flatten.numel()] = weights_flatten
        pos_idx_topk = torch.topk(extended_weights_flatten.where(extended_weights_flatten > 0,
                    torch.zeros(extended_weights_flatten.shape)).view(-1, group_size).abs(), k=k_pos)[1]
        neg_idx_topk = torch.topk(extended_weights_flatten.where(extended_weights_flatten < 0,
                    torch.zeros(extended_weights_flatten.shape)).view(-1, group_size).abs(), k=k_neg)[1]
        pos_weight_values = extended_weights_flatten.where(extended_weights_flatten > 0, torch.zeros(
            extended_weights_flatten.shape)).view(-1, group_size).gather(dim=1, index=pos_idx_topk).mean(dim=0)
        neg_weight_values = extended_weights_flatten.where(extended_weights_flatten < 0, torch.zeros(
            extended_weights_flatten.shape)).view(-1, group_size).gather(dim=1, index=neg_idx_topk).mean(dim=0)
    quantized_weight_values = torch.cat([pos_weight_values, neg_weight_values], dim=0)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.set_quantized_weight_values(quantized_weight_values)


@torch.no_grad()
def update_masks_globally(model, params_prime, amount=0.5):
    weights = []
    for weight in params_prime:
        weights.append(weight.clone().detach().to(device))
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
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.set_mask(masks[mask_idx])
            mask_idx += 1


@torch.no_grad()
def update_masks_unstructured(model, amount=0.5):
    weights = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
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
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.set_mask(masks[mask_idx])
            mask_idx += 1


@torch.no_grad()
def update_mask_structured(layer, group_size=16, amount=0.9):
    thr = int(group_size * (1 - amount))
    if layer.weight.numel() % group_size == 0:
        col_idx = torch.topk(layer.weight.view(-1, group_size).abs(), k=thr)[1].view(-1)
        row_idx = torch.arange(layer.weight.view(-1, group_size).shape[0]).to(device).repeat_interleave(thr) * group_size
        idx = col_idx + row_idx
        mask = torch.zeros(layer.weight.shape).to(device)
        mask.view(-1)[idx] = 1
        layer.set_mask(mask)
    else:
        n_row = layer.weight.numel() // group_size
        num_rest = layer.weight.numel() % group_size
        thr_rest = int(num_rest * (1 - amount))
        weight_dividable = layer.weight.view(-1)[:n_row * group_size].view(-1, group_size)
        col_idx_dividable = torch.topk(weight_dividable.abs(), k=thr)[1].view(-1)
        row_idx_dividable = torch.arange(n_row).to(device).repeat_interleave(thr) * group_size
        idx_dividable = col_idx_dividable + row_idx_dividable
        weight_rest = layer.weight.view(-1)[n_row * group_size:]
        col_idx_rest = torch.topk(weight_rest.abs(), k=thr_rest)[1].view(-1)
        row_idx_rest = torch.tensor(n_row + 1).to(device).repeat_interleave(thr) * group_size
        idx_rest = col_idx_rest + row_idx_rest
        idx = torch.cat([idx_dividable, idx_rest], dim=0)
        mask = torch.zeros(layer.weight.shape).to(device)
        mask.view(-1)[idx] = 1
        layer.set_mask(mask)


@torch.no_grad()
def prune_weight_abs(param, amount=0.9):
    thr = int(param.numel() * amount)
    idx = torch.argsort(param.view(-1).abs())[:thr]
    param.view(-1)[idx] = 0


@torch.no_grad()
def prune_weight_structured_abs(param, group_size=16, amount=0.9):
    thr = int(group_size * amount)
    if param.numel() % group_size == 0:
        col_idx = torch.argsort(param.view(-1, group_size).abs())[:, :thr].reshape(-1)
        row_idx = torch.arange(param.view(-1, group_size).shape[0]).to(device).repeat_interleave(thr) * group_size
        idx = col_idx + row_idx
        param.view(-1)[idx] = 0
    else:
        n_row = param.numel() // group_size
        param_dividable = param.view(-1)[:n_row * group_size].view(-1, group_size)
        col_idx_dividable = torch.argsort(param_dividable.abs())[:, :thr].reshape(-1)
        row_idx_dividable = torch.arange(param_dividable.shape[0]).to(device).repeat_interleave(thr) * group_size
        idx_dividable = col_idx_dividable + row_idx_dividable
        param.view(-1)[idx_dividable] = 0
        n_rest = param.numel() % group_size
        thr_rest = int(n_rest * amount)
        param_rest = param.view(-1)[n_row * group_size:]
        idx_rest = torch.argsort(param_rest.abs())[:thr_rest]
        param_rest[idx_rest] = 0


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
