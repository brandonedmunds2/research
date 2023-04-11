import torch
from torch import optim
from opacus import PrivacyEngine
from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed, _generate_noise
from typing import List, Union
from opt_einsum.contract import contract

class MaskedDPOptimizer(DPOptimizer):
    def __init__(self, masks, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masks=masks
    def _get_flat_grad_sample(self, p: torch.Tensor):
        """
        Return parameter's per sample gradients as a single tensor.
        By default, per sample gradients (``p.grad_sample``) are stored as one tensor per
        batch basis. Therefore, ``p.grad_sample`` is a single tensor if holds results from
        only one batch, and a list of tensors if gradients are accumulated over multiple
        steps. This is done to provide visibility into which sample belongs to which batch,
        and how many batches have been processed.
        This method returns per sample gradients as a single concatenated tensor, regardless
        of how many batches have been accumulated
        Args:
            p: Parameter tensor. Must have ``grad_sample`` attribute
        Returns:
            ``p.grad_sample`` if it's a tensor already, or a single tensor computed by
            concatenating every tensor in ``p.grad_sample`` if it's a list
        Raises:
            ValueError
                If ``p`` is missing ``grad_sample`` attribute
        """

        if not hasattr(p, "grad_sample"):
            raise ValueError(
                "Per sample gradient not found. Are you using GradSampleModule?"
            )
        if p.grad_sample is None:
            raise ValueError(
                "Per sample gradient is not initialized. Not updated in backward pass?"
            )
        if isinstance(p.grad_sample, torch.Tensor):
            ret = p.grad_sample
        elif isinstance(p.grad_sample, list):
            ret = torch.cat(p.grad_sample, dim=0)
        else:
            raise ValueError(f"Unexpected grad_sample type: {type(p.grad_sample)}")

        return ret
    def clip_and_accumulate(self):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """

        if len(self.grad_samples[0]) == 0:
            # Empty batch
            per_sample_clip_factor = torch.zeros((0,))
        else:
            per_param_norms = [
                g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
            ]
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            per_sample_clip_factor = (
                self.max_grad_norm / (per_sample_norms + 1e-6)
            ).clamp(max=1.0)

        for i,p in enumerate(self.params):
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            grad = contract("i,i...", per_sample_clip_factor, grad_sample)

            if(i in self.masks):
                grad.flatten().scatter_(dim=0,index=torch.nonzero(
                        self.masks[i].flatten()
                    ).flatten(),src=p.grad.flatten())

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)
    def add_noise(self):
        for i,p in enumerate(self.params):
            _check_processed_flag(p.summed_grad)

            noise = _generate_noise(
                std=self.noise_multiplier * self.max_grad_norm,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            if(i in self.masks):
                noise.mul_(torch.logical_not(self.masks[i]).to(torch.int32))
            p.grad = (p.summed_grad + noise).view_as(p)

            _mark_as_processed(p.summed_grad)

class MaskedPrivacyEngine(PrivacyEngine):
    def __init__(self, masks, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masks=masks
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
            masks=self.masks,
        )
    
if __name__ == "__main__":
    pass