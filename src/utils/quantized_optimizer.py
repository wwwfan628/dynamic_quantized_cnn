from torch.optim.optimizer import Optimizer
import torch
import math
import numpy as np
from ._quant_optim_functional import quant_sgd, quant_adam

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Quant_SGD(Optimizer):
    def __init__(self, params, lr=0.4, momentum=0, dampening=0, weight_decay=0, nesterov=False, params_prime=None,
                 group_size=64, num_values=16, update_available_values=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if params_prime is None:
            raise ValueError("Invalid params_prime value: {}".format(params_prime))
        else:
            params_prime = list(param_prime.clone().detach().to(device) for param_prime in params_prime)
        if num_values <= 0 or num_values % 2 != 0:
            raise ValueError("Invalid num_values value: {}".format(num_values))
        if group_size <= 0:
            raise ValueError("Invalid group_size value: {}".format(group_size))
        # initialize available_values
        available_values = torch.zeros(num_values).to(device)
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov,
                        params_prime=params_prime, group_size=group_size, num_values=num_values,
                        available_values=available_values, update_available_values=update_available_values)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Quant_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Quant_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            params_prime_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            num_values = group['num_values']
            group_size = group['group_size']
            available_values = group['available_values']
            update_available_values = group['update_available_values']
            lr = group['lr']

            for p_ind, p in enumerate(group['params']):
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    params_prime_with_grad.append(group['params_prime'][p_ind])

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            quant_sgd(params_with_grad, d_p_list, momentum_buffer_list, weight_decay, momentum, lr, dampening,
                      nesterov, params_prime_with_grad, available_values, group_size, num_values, update_available_values)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss



class Quant_Adam(Optimizer):
    """Implements Adam algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing running averages of gradient and its
            square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this algorithm from the paper
            `On the Convergence of Adam and Beyond`_(default: False)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, params_prime=None, perm_size=16):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if params_prime is None:
            raise ValueError("Invalid params_prime value: {}".format(params_prime))
        else:
            params_prime = list(param_prime.clone().detach().to(device) for param_prime in params_prime)
        # compute params_sorted
        params_sorted = []
        for param_prime in params_prime:
            if param_prime.numel() % perm_size == 0:
                params_sorted.append(torch.sort(param_prime.view(-1, perm_size)))
            else:
                n_row = math.ceil(param_prime.numel() / perm_size)
                param_sorted = torch.zeros([n_row, perm_size]).to(device)
                param_sorted.view(-1)[:param_prime.numel()] = param_prime.clone().detach().view(-1).to(device)
                param_sorted.view(-1)[param_prime.numel():] = float('inf')
                params_sorted.append(torch.sort(param_sorted)[0])
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, params_prime=params_prime, params_sorted=params_sorted, perm_size=perm_size)
        super(Perm_Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Perm_Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            params_prime_with_grad = []
            params_sorted_with_grad = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []

            for p_ind, p in enumerate(group['params']):
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)
                    params_prime_with_grad.append(group['params_prime'][p_ind])
                    params_sorted_with_grad.append(group['params_sorted'][p_ind])

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            perm_adam(params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
                            group['amsgrad'], beta1, beta2, group['lr'], group['weight_decay'], group['eps'],
                            params_prime_with_grad, params_sorted_with_grad, group['perm_size'])
        return loss
