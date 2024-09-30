from typing import Dict
import torch
import torch.nn as nn
from .base import HeadModule


class NeuralClassifierHead(HeadModule):
    def __init__(self, classifier: nn.Module, tgt_ind: int) -> None:
        super().__init__()
        self.classifier = classifier
        self.tgt_ind = tgt_ind

    def state_dict(self) -> Dict[str, torch.Tensor]:
        state = {
            "classifier": self.classifier.state_dict(),
            "tgt_ind": self.tgt_ind,
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.classifier.load_state_dict(state_dict["classifier"])
        self.tgt_ind = state_dict["tgt_ind"]

    def get_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        logits: torch.Tensor = self.classifier(x)

        criterion = nn.CrossEntropyLoss(**kwargs)
        loss = criterion(
            logits,
            logits.new_ones(size=(logits.size(0),), dtype=torch.long) * self.tgt_ind,
        )
        return loss
