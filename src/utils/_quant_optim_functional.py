from typing import List, Optional
import math
import torch
import numpy as np
from torch import Tensor

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def quant_sgd(params: List[Tensor], d_p_list: List[Tensor], momentum_buffer_list: List[Optional[Tensor]],
              weight_decay: float, momentum: float, lr: float, dampening: float, nesterov: bool,
              params_prime: List[Tensor], available_values: List[Tensor], group_size: int, num_values: int,
              update_available_values: bool):
    """
    Functional API that performs Perm SGD algorithm computation.
    """
    for i, (param, param_prime) in enumerate(zip(params, params_prime)):
        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        # param.add_(d_p, alpha=-lr)  # sgd, from pytorch original code
        param_prime.add_(d_p, alpha=-lr)    # update unconstrained weight

    if update_available_values:
        num_pos = int(0.5 * num_values)
        num_neg = int(0.5 * num_values)
        params_prime_flatten = np.zeros(0)
        for param_prime in params_prime:
            params_prime_flatten = np.append(params_prime_flatten, param_prime.view(-1).clone().detach().cpu())
        params_prime_flatten = torch.Tensor(params_prime_flatten).to(device)
        if params_prime_flatten.numel() % group_size == 0:
            pos_idx_topk = torch.topk(params_prime_flatten.where(params_prime_flatten > 0, torch.zeros(
                params_prime_flatten.shape)).view(-1, group_size).abs(), k=num_pos)[1]
            neg_idx_topk = torch.topk(params_prime_flatten.where(params_prime_flatten < 0, torch.zeros(
                params_prime_flatten.shape)).view(-1, group_size).abs(), k=num_neg)[1]
            pos_weight_values = params_prime_flatten.where(params_prime_flatten > 0,
                torch.zeros(params_prime_flatten.shape)).view(-1, group_size).gather(dim=1, index=pos_idx_topk).mean(dim=0)
            neg_weight_values = params_prime_flatten.where(params_prime_flatten < 0,
                torch.zeros(params_prime_flatten.shape)).view(-1, group_size).gather(dim=1, index=neg_idx_topk).mean(dim=0)
        else:
            n_row = math.ceil(params_prime_flatten.numel() / group_size)
            extended_params_prime_flatten = torch.zeros(n_row * group_size)
            extended_params_prime_flatten[:params_prime_flatten.numel()] = params_prime_flatten
            pos_idx_topk = torch.topk(extended_params_prime_flatten.where(extended_params_prime_flatten > 0,
                torch.zeros(extended_params_prime_flatten.shape)).view(-1, group_size).abs(), k=num_pos)[1]
            neg_idx_topk = torch.topk(extended_params_prime_flatten.where(extended_params_prime_flatten < 0,
                torch.zeros(extended_params_prime_flatten.shape)).view(-1, group_size).abs(), k=num_neg)[1]
            pos_weight_values = extended_params_prime_flatten.where(extended_params_prime_flatten > 0,
                torch.zeros(extended_params_prime_flatten.shape)).view(-1, group_size).gather(dim=1,
                index=pos_idx_topk).mean(dim=0)
            neg_weight_values = extended_params_prime_flatten.where(extended_params_prime_flatten < 0,
                torch.zeros(extended_params_prime_flatten.shape)).view(-1, group_size).gather(dim=1,
                index=neg_idx_topk).mean(dim=0)
        available_values.view(-1)[:] = torch.cat([pos_weight_values, neg_weight_values], dim=0)

    for i, (param, param_prime) in enumerate(zip(params, params_prime)):
        idx = torch.argmin(
            (param_prime.view(-1).unsqueeze(1).expand(-1, available_values.shape[0]) - available_values).abs(),
            dim=1).unsqueeze(0)
        param.view(-1)[:] = (available_values.unsqueeze(0).expand(param.view(-1).shape[0], -1)).gather(-1, idx)



def quant_adam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor],
              max_exp_avg_sqs: List[Tensor], state_steps: List[int], amsgrad: bool, beta1: float, beta2: float,
              lr: float, weight_decay: float, eps: float, params_prime: List[Tensor], params_sorted: List[Tensor], perm_size):
    """
    Functional API that performs Adam algorithm computation.
    """
    for i, (param, param_prime, param_sorted) in enumerate(zip(params, params_prime, params_sorted)):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1
        # param.addcdiv_(exp_avg, denom, value=-step_size)   # adam, from pytorch original code
        param_prime.addcdiv_(exp_avg, denom, value=-step_size)
        if param_prime.numel() % perm_size == 0:
            # reshape and then sort param_prime along dim=1
            idx_tmp = torch.argsort(param_prime.view(-1, perm_size)).view(-1)
            # row indices
            row_idx = perm_size * torch.arange(param_prime.view(-1, perm_size).size(0)).repeat_interleave(perm_size).to(
                device)
            # assign values
            param.view(-1)[idx_tmp + row_idx] = param_sorted.view(-1)
        else:
            n_row = math.ceil(param_prime.numel() / perm_size)
            param_prime_tmp = torch.zeros([n_row, perm_size]).to(device)
            param_prime_tmp.view(-1)[:param_prime.numel()] = param_prime.clone().detach().view(-1).to(device)
            param_prime_tmp.view(-1)[param_prime.numel():] = float('inf')
            # reshape and then sort param_prime_tmp along dim=1
            idx_tmp = torch.argsort(param_prime_tmp.view(-1, perm_size)).view(-1)[:param_prime.numel()]
            # row indices
            row_idx = perm_size * torch.arange(param_prime_tmp.view(-1, perm_size).size(0)).repeat_interleave(perm_size).to(device)[:param_prime.numel()]
            # assign values
            param.view(-1)[idx_tmp + row_idx] = param_sorted.view(-1)[:param_prime.numel()]
