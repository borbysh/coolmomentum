import torch
from torch.optim.optimizer import Optimizer, required

class Coollin(Optimizer):
    """
        lr (float): learning rate
        momentum (float, optional): initinal momentum constant (0 for SGD)
        weight_decay (float, optional): weight decay (L2 penalty) 
        beta: cooling rate, close to 1, if beta=1 then no cooling
    """

    def __init__(self, params, lr=0.01, rho_0=0.99, cool_steps=78200, # of simulated annealing iterations
                 weight_decay=0.0, dropout=0.0):

        defaults = dict(lr=lr, rho_0=rho_0, cool_steps=cool_steps, weight_decay=weight_decay, dropout=dropout)
        super(Coollin, self).__init__(params, defaults)
        self.T = 0.0
        self.number = 0
        self.epoch = None
        self.rho = 0.0
        self.filename='temperature_cool_lin(lr={}'.format(lr)+';rho_0={}'.format(rho_0)+';cool_steps={}'.format(cool_steps)+').txt'
        self.period = 391
        #self.beta = ((1 - rho_0)/(1 + rho_0))**(1/cool_steps)
 

    def __setstate__(self, state):
        super(Coollin, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Sparse gradients are not supported')

                state = self.state[p]
                
                # mask
                #m = torch.ones_like(p.data) * group['dropout']
                #mask = torch.bernoulli(m)

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of parameter update values
                    state['dx'] = torch.zeros_like(p.data)
                
                dx = state['dx']
                rho_0 = group['rho_0']
                cool_steps = group['cool_steps']
                i = state['step']
                
                self.rho = rho_0 * (1 - i/cool_steps) / abs(1 - rho_0 * i/cool_steps)
                self.rho = max(self.rho, 0)
                lr_dropout = group['lr']*(1+self.rho)/2 # lrn instead of lr
                #lr_dropout = lr_dropout * mask
                
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha = group['weight_decay'])
                    
                dx.mul_(self.rho).add_(grad, alpha=lr_dropout)
                
                self.T += torch.sum(torch.mul(dx, dx))  #squared step
                self.number += torch.sum(torch.ones_like(p.data)) # of trainable params
                

                p.data.add_(dx, alpha = -1)
                

        if state['step']%self.period==0:
            self.epoch = (state['step'] // self.period) - 1 
            self.write_log()        
                   
        return loss

    def write_log(self):

        self.T /= self.number
        if self.epoch==0:
            tfile=open(self.filename, "w")
            print('#e: {}, rho: {}, train temperature: {}', "\r", file=tfile )
        else:
            tfile=open(self.filename, "a")       
        print(self.epoch, self.rho, self.T.cpu().numpy(), file=tfile)
        tfile.close()
        self.T = 0.0
        self.number = 0
        return None 

