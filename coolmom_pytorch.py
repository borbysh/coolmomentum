import torch
from torch.optim.optimizer import Optimizer, required

class SGD(Optimizer):
    """
        lr (float): learning rate
        momentum (float, optional): initinal momentum constant (0 for SGD)
        weight_decay (float, optional): weight decay (L2 penalty) 
        beta: cooling rate, close to 1, if beta=1 then no cooling
    """

    def __init__(self, params, lr=required, momentum=0,
                 weight_decay=0,  beta=1.0):

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, beta=beta, beta_power=beta)
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)

    @torch.no_grad()
    def step(self):

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            beta=group['beta']
            beta_power=group['beta_power']
            group['beta_power']=group['beta_power']*beta
            ro = 1-(1 - momentum)/beta_power
            ro= max(ro,0) #ro instead of momentum
            lrn=group['lr']*(1+ro)/2 # lrn instead of lr 

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf *= 0
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(ro).add_(d_p, alpha=-lrn)
                d_p=buf
                p.add_(d_p)

        return None
