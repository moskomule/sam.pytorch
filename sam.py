import warnings
from typing import Iterable

import torch
from torch.optim._multi_tensor import SGD

__all__ = ["SAMSGD"]


class SAMSGD(torch.optim.Optimizer):
    """ SGD wrapped with Sharp-Aware Minimization

    Args:
        params: tensors to be optimized
        lr: learning rate
        momentum: momentum factor
        dampening: damping factor
        weight_decay: weight decay factor
        nesterov: enables Nesterov momentum
        rho: neighborhood size

    """

    def __init__(self,
                 params: Iterable[torch.Tensor],
                 lr: float,
                 momentum: float = 0,
                 dampening: float = 0,
                 weight_decay: float = 0,
                 nesterov: bool = False,
                 rho: float = 0.05,
                 ):
        if rho <= 0:
            raise ValueError(f"Invalid neighborhood size: {rho}")
        defaults = dict(rho=rho, lr=lr)
        params = list(params)
        self._internal_optim = SGD(params, lr=lr, momentum=momentum, dampening=dampening,
                                   weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)
        if len(self.param_groups) > 1:
            warnings.warn("The computation cost of this implementation depends on the number of param_groups. "
                          "Also, gradient_norm is computed group-wise, which may incur unexpected behavior.")

    @torch.no_grad()
    def step(self,
             closure
             ) -> torch.Tensor:
        """

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns: the loss value evaluated on the original point

        """
        closure = torch.enable_grad()(closure)
        loss = closure().detach()

        for group, in_group in zip(self.param_groups, self._internal_optim.param_groups):
            grads = []
            params_with_grads = []

            rho = group['rho']
            # update internal_optim's learning rate
            in_group['lr'] = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    grads.append(p.grad)
                    params_with_grads.append(p)
            device = grads[0].device

            # compute \hat{\epsilon}=\rho/\norm{g}\|g\|
            grad_norm = torch.stack([g.detach().norm(2).to(device) for g in grads]).norm(2)
            epsilon = grads  # alias for readability
            torch._foreach_mul_(epsilon, rho / grad_norm)

            # virtual step toward \epsilon
            torch._foreach_add_(params_with_grads, epsilon)
            # compute g=\nabla_w L_B(w)|_{w+\hat{\epsilon}}
            closure()
            # virtual step back to the original point
            torch._foreach_sub_(params_with_grads, epsilon)

        self._internal_optim.step()
        return loss

    def state_dict(self) -> dict:
        state_dict = super().state_dict()
        state_dict["internal"] = self._internal_optim.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        _internal = state_dict.pop("internal")
        self._internal_optim.load_state_dict(_internal)
        super().load_state_dict(state_dict)
