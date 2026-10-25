"""Published reset-after GRU architecture and TensorFlow epsilon-hat Adam semantics."""
import torch

class KerasAdam(torch.optim.Optimizer):
    def __init__(self,params,lr=.02,eps=.1):
        super().__init__(params,dict(lr=lr,eps=eps,iterations=0))

    @torch.no_grad()
    def step(self,closure=None):
        for group in self.param_groups:
            group['iterations']+=1
            for p in group['params']:
                # Original traced switch_case supplies zero gradients for inactive adapters.
                grad=p.grad if p.grad is not None else torch.zeros_like(p)
                s=self.state[p]
                if not s:
                    s.update(step=0,m=torch.zeros_like(p),v=torch.zeros_like(p))
                s['step']=group['iterations']
                b1=torch.tensor(.9,dtype=p.dtype,device=p.device)
                b2=torch.tensor(.999,dtype=p.dtype,device=p.device)
                s['m'].mul_(b1).add_(grad*(1-b1))
                s['v'].mul_(b2).add_(grad.square()*(1-b2))
                rate=group['lr']*torch.sqrt(1-b2**s['step'])/(1-b1**s['step'])
                p.add_(-rate*s['m']/(s['v'].sqrt()+group['eps']))

def swap_gates(x,axis):
    z,r,h=torch.chunk(x,3,dim=axis)
    return torch.cat((r,z,h),dim=axis)

class PublishedGRU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=torch.nn.ModuleList(torch.nn.GRU(8192 if i==0 else 1024,1024,batch_first=True) for i in range(5))
        self.head=torch.nn.Linear(1024,41)
        self.h0=torch.nn.Parameter(torch.zeros(1,1024))
        self.adapters=torch.nn.ModuleList(torch.nn.Linear(256,256) for _ in range(19))

    def load_tf(self,fixture):
        with torch.no_grad():
            for i,l in enumerate(self.layers):
                k,r,b=[torch.tensor(fixture[f'initial_{3*i+j:03d}']) for j in range(3)]
                l.weight_ih_l0.copy_(swap_gates(k,1).T)
                l.weight_hh_l0.copy_(swap_gates(r,1).T)
                l.bias_ih_l0.copy_(swap_gates(b[0],0))
                l.bias_hh_l0.copy_(swap_gates(b[1],0))
            self.head.weight.copy_(torch.tensor(fixture['initial_015']).T)
            self.head.bias.copy_(torch.tensor(fixture['initial_016']))
            self.h0.copy_(torch.tensor(fixture['initial_017']))
            for a in self.adapters:
                a.weight.copy_(torch.eye(256)); a.bias.zero_()
            self.adapters[0].weight.copy_(torch.tensor(fixture['initial_018']).T)
            self.adapters[0].bias.copy_(torch.tensor(fixture['initial_019']))

    def forward(self,x,day=0):
        x=torch.nn.functional.softsign(self.adapters[day](x))
        x=torch.nn.functional.dropout(x,.2,self.training)
        x=x.unfold(1,32,4).permute(0,1,3,2).flatten(2)
        for i,l in enumerate(self.layers):
            if self.training:
                mask=torch.nn.functional.dropout(torch.ones_like(x[:,:1]),.4,True)
                x=x*mask
            h=self.h0.unsqueeze(0).expand(1,x.shape[0],-1).contiguous() if i==0 else None
            x,_=l(x,h)
        return self.head(x)

    def regularization(self,day=0):
        return 1e-5*(sum(l.weight_ih_l0.square().sum() for l in self.layers)+self.adapters[day].weight.square().sum())

    def tf_arrays(self,gradient=False):
        def get(p): return p.grad if gradient else p.detach()
        arrays=[]
        for l in self.layers:
            arrays.extend((swap_gates(get(l.weight_ih_l0).T,1),swap_gates(get(l.weight_hh_l0).T,1),
                           torch.stack((swap_gates(get(l.bias_ih_l0),0),swap_gates(get(l.bias_hh_l0),0)))))
        arrays.extend((get(self.head.weight).T,get(self.head.bias),get(self.h0),get(self.adapters[0].weight).T,get(self.adapters[0].bias)))
        return [a.detach().cpu().numpy().copy() for a in arrays]
