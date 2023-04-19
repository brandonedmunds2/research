import torch
from torch import optim
from opacus import PrivacyEngine
from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed, _generate_noise
from typing import List, Union
from torch.nn.utils.prune import _compute_nparams_toprune, _validate_pruning_amount

class MaskedDPOptimizer(DPOptimizer):
    def __init__(self, amount,strategy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.amount=amount
        self.strategy=strategy
        self.prev_mask={}
    def compute_mask(self,t,amount,strategy,param_ind):
        default_mask=torch.ones_like(t)
        tensor_size = t.nelement()
        nparams_toprune = _compute_nparams_toprune(amount, tensor_size)
        _validate_pruning_amount(nparams_toprune, tensor_size)
        mask = default_mask.clone(memory_format=torch.contiguous_format)
        if nparams_toprune != 0:
            if(strategy=='random'):
                prob = torch.rand_like(t)
                topk = torch.topk(prob.view(-1), k=nparams_toprune)
            elif(strategy=='magnitude'):
                topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=False)
            elif(strategy=='magnitude reverse'):
                topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=True)
            elif(strategy=='rolling'):
                if(param_ind not in self.prev_mask):
                    prob = torch.rand_like(t)
                    topk = torch.topk(prob.view(-1), k=nparams_toprune)
                    mask.view(-1)[topk.indices] = 0
                else:
                    mask=torch.roll(self.prev_mask[param_ind],1)
                self.prev_mask[param_ind]=mask
                return mask
            mask.view(-1)[topk.indices] = 0
        return mask
    def add_noise(self):
        for i,p in enumerate(self.params):
            _check_processed_flag(p.summed_grad)

            noise = _generate_noise(
                std=self.noise_multiplier * self.max_grad_norm,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            mask=self.compute_mask(p,amount=self.amount,strategy=self.strategy,param_ind=i)
            noise.mul_(mask)
            p.grad = (p.summed_grad + noise).view_as(p)

            _mark_as_processed(p.summed_grad)

class MaskedPrivacyEngine(PrivacyEngine):
    def __init__(self, amount,strategy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.amount=amount
        self.strategy=strategy
    def _prepare_optimizer(
        self,
        optimizer: optim.Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        expected_batch_size: int,
        loss_reduction: str = "mean",
        distributed: bool = False,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode="hooks",
    ) -> DPOptimizer:
        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        return MaskedDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=self.secure_mode,
            amount=self.amount,
            strategy=self.strategy
        )
    
if __name__ == "__main__":
    pass