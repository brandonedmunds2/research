import torch
import torch.nn.utils.prune as prune
from torch.nn.utils.prune import L1Unstructured, _compute_nparams_toprune, _validate_pruning_amount

class CustomL1Unstructured(L1Unstructured):
    def __init__(self, largest, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.largest=largest
    def compute_mask(self, t, default_mask):
        tensor_size = t.nelement()
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        _validate_pruning_amount(nparams_toprune, tensor_size)
        mask = default_mask.clone(memory_format=torch.contiguous_format)
        if nparams_toprune != 0:
            topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=self.largest)
            mask.view(-1)[topk.indices] = 0
        return mask

def pruner(model,type,layers,amount,largest=False):
    params=[]
    for name,module in model.named_modules():
        if(name in layers and hasattr(module,"weight")):
            params.append((module,"weight"))
    if(type=="random"):
        prune.global_unstructured(params,pruning_method=prune.RandomUnstructured,amount=amount)
    elif(type=="magnitude"):
        prune.global_unstructured(params,pruning_method=CustomL1Unstructured,largest=largest,amount=amount)

def end_prune(model):
    for name,module in model.named_modules():
        if(hasattr(module,"weight")):
            if(prune.is_pruned(module)):
                prune.remove(module,"weight")

def prune_grads(optimizer,masks):
    if(masks==None):
        return
    for key, val in masks.items():
        optimizer.grad_samples[key].mul_(val)

def get_masks(model,n_pd):
    masks={}
    # assume bias before weights and bias not pruned and order
    for i,n_p in enumerate(model.named_parameters()):
        key=n_p[0]+"_mask"
        if(key in n_pd):
            masks[i]=n_pd[key]
    return masks

def prune_mask(model,type,layers,amount,largest):
    pruner(model,type,layers,amount,largest)
    n_pd=dict(model.named_buffers())
    end_prune(model)
    masks=get_masks(model,n_pd)
    return masks

if __name__ == "__main__":
    pass