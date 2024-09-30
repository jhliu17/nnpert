import torch
import torch.nn as nn

from typing import Dict
from abc import abstractmethod


class PertModule(nn.Module):
    @abstractmethod
    def perturb(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_pert_ind(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_pert_val(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def update(self, x: torch.Tensor, y: torch.Tensor = None, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        raise NotImplementedError

    def to(self, device: torch.device):
        super().to(device)
        return self
