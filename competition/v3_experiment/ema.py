"""EMA is updated only after a successful optimizer update, after warm-up."""
import copy
import torch


class ModelEMA:
    def __init__(self,model,decay=.999):
        self.model=copy.deepcopy(model).eval()
        self.decay=float(decay)
        self.updates=0
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self,model,optimizer_succeeded=True):
        if not optimizer_succeeded:return
        current=model.state_dict()
        for name,value in self.model.state_dict().items():
            if value.is_floating_point():value.mul_(self.decay).add_(current[name].detach(),alpha=1-self.decay)
            else:value.copy_(current[name])
        self.updates+=1

    def state_dict(self):
        return {'model':self.model.state_dict(),'decay':self.decay,'updates':self.updates}

    def load_state_dict(self,state):
        if state['decay']!=self.decay:raise ValueError('EMA decay changed on resume')
        self.model.load_state_dict(state['model'],strict=True)
        self.updates=state['updates']
