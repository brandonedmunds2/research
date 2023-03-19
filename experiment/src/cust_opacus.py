from torch import optim
from opacus import PrivacyEngine
from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed, _generate_noise
from typing import List, Union

class CustomDPOptimizer(DPOptimizer):
    def __init__(self, masks, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masks=masks
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
                noise.mul_(self.masks[i])
            p.grad = (p.summed_grad + noise).view_as(p)

            _mark_as_processed(p.summed_grad)

class CustomPrivacyEngine(PrivacyEngine):
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

        return CustomDPOptimizer(
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