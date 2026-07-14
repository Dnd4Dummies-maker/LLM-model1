import math

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def get_optimizer(model, lr: float = 3e-4, weight_decay: float = 0.1, betas: tuple = (0.9, 0.95)):
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)


def get_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
