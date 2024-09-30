from typing import Dict
import torch
import torch.nn as nn
import numpy as np

from .base import HeadModule


class ValueIndexerHead(HeadModule):
    def __init__(self, ind: int, increase_value: bool = True):
        super().__init__()
        self.ind = ind
        self.increase_value = increase_value

    def state_dict(self) -> Dict[str, torch.Tensor]:
        state = {
            "ind": self.ind,
            "increase_value": self.increase_value,
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.ind = state_dict["ind"]
        self.increase_value = state_dict["increase_value"]

    def get_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        ind_x = torch.index_select(
            x, dim=1, index=x.new_ones(1, dtype=torch.long) * self.ind
        ).squeeze(dim=-1)

        loss = ind_x * (-1 if self.increase_value else 1)
        if kwargs.get("reduction", "mean") == "none":
            pass
        else:
            loss = loss.mean()
        return loss
