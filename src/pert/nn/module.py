import torch
import torch.nn as nn
import numpy as np

from typing import Dict
from abc import abstractmethod

from .op import (
    BatchTriggerPerturbationFunction,
    SequenceTriggerPerturbationFunction,
    SequenceEmbeddingTriggerPerturbationFunction,
    initial_random_ind,
    look_up_category_embedding,
)

EPSILON = np.finfo(np.float32).tiny


class BaseTriggerPerturbation(nn.Module):
    def __init__(
        self,
        pert_num: int,
        length: int,
        init_pert_ind: torch.LongTensor = None,
        init_pert_val: torch.Tensor = None,
    ):
        super().__init__()
        if pert_num > length:
            raise ValueError("Perturbation number is larger than the total length.")
        if length != init_pert_val.shape[-1]:
            raise ValueError("Perturbation value is incompactible the total length.")

        self.pert_num = pert_num
        self.length = length

        # if init_pert_vec is not given, we assume it would be loaded from a state dict
        self.register_buffer(
            "pert_ind",
            init_pert_ind.detach().clone()
            if init_pert_ind is not None
            else torch.LongTensor(initial_random_ind(pert_num, length)),
        )
        self.register_buffer(
            "pert_val",
            init_pert_val.detach().clone().float()
            if init_pert_val is not None
            else torch.ones((self.length,)),
        )

    @abstractmethod
    def forward(self, logits: torch.Tensor, tau: float = 1.0):
        raise NotImplementedError


class TriggerPerturbation(BaseTriggerPerturbation):
    def __init__(
        self,
        pert_num: int,
        length: int,
        init_pert_ind: torch.LongTensor = None,
        init_pert_val: torch.Tensor = None,
    ):
        super().__init__(pert_num, length, init_pert_ind, init_pert_val)

        self.weight = nn.Parameter(torch.zeros((pert_num, length)))

    def forward(self, x: torch.Tensor):
        pert_x = SequenceTriggerPerturbationFunction.apply(
            x, self.pert_ind, self.pert_val, self.weight
        )
        return pert_x

    @torch.inference_mode()
    def trial(self, x: torch.Tensor, pert_vecs: torch.Tensor):
        pert_x = BatchTriggerPerturbationFunction.apply(x, pert_vecs, self.pert_val)
        return pert_x


class EmbeddingTriggerPerturbation(TriggerPerturbation):
    def __init__(
        self,
        pert_num: int,
        length: int,
        embedding_module: nn.Module,
        init_pert_ind: torch.LongTensor = None,
        init_pert_val: torch.Tensor = None,
    ):
        super().__init__(pert_num, length, init_pert_ind, init_pert_val)
        self.register_buffer("embed_weight", embedding_module.weight.detach().clone())

    def forward(self, x: torch.Tensor):
        pert_x_embed = SequenceEmbeddingTriggerPerturbationFunction.apply(
            x, self.pert_ind, self.pert_val, self.embed_weight, self.weight
        )
        return pert_x_embed

    @torch.inference_mode()
    def trial(self, x: torch.Tensor, pert_vecs: torch.Tensor):
        pert_x = BatchTriggerPerturbationFunction.apply(x, pert_vecs, self.pert_val)

        pert_x_embed = look_up_category_embedding(pert_x, self.embed_weight)
        return pert_x_embed
