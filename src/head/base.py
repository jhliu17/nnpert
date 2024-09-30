import torch
import torch.nn as nn

from typing import Dict
from abc import abstractmethod


class HeadModule(nn.Module):
    @abstractmethod
    def get_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        raise NotImplementedError
