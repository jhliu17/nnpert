import torch
import torch.nn as nn
import numpy as np
import sklearn

from functools import partial
from sklearn.utils import check_random_state
from lime.lime_base import LimeBase
from tqdm import tqdm
from typing import Callable, Dict, List, Literal, Union
from dataclasses import dataclass, field
from ..base import PertModule
from ..nn.module import (
    BaseTriggerPerturbation,
    TriggerPerturbation,
    EmbeddingTriggerPerturbation,
)
from ..utils.mask import MaskInput, RandomlyMaskInput


@dataclass
class LIMEPertModuleConfig:
    mask_type: Literal["random"] = "random"

    batch_size: int = 128

    masked_batch_size: int = 128

    neighbor_size: int = 128

    neighbor_feat_num: int = 100000

    feature_selection: str = "auto"

    random_seed: int = 2024

    kernel_width: float = 0.25

    distance_metric: str = "cosine"

    # trigger pert type
    trigger_pert_type: str = "trigger_perturbation"


class LIMEPertModule(PertModule):
    def __init__(
        self,
        pert_num: int,
        pert_space: int,
        pert_val: torch.Tensor,
        embedding_module: torch.nn.Module = None,
        mask_type: str = "image",
        batch_size: int = 128,
        neighbor_size: int = 128,
        masked_batch_size: int = 128,
        random_seed: int = 2024,
        kernel_width: float = 0.25,
        distance_metric: str = "cosine",
        neighbor_feat_num: int = 100000,
        feature_selection: str = "auto",
        init_pert_ind: List[int] = [],
        **kwargs,
    ) -> None:
        super().__init__()
        self.pert_val = pert_val
        self.pert_num = pert_num
        self.mask_type = mask_type
        self.batch_size = batch_size
        self.masked_batch_size = masked_batch_size
        self.neighbor_size = neighbor_size
        self.distance_metric = distance_metric
        self.neighbor_feat_num = neighbor_feat_num
        self.feature_selection = feature_selection

        self.random_seed = random_seed
        self.random_state = check_random_state(random_seed)
        kernel_width = float(kernel_width)

        def kernel(d, kernel_width):
            return np.sqrt(np.exp(-(d**2) / kernel_width**2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)
        self.lime_model = LimeBase(
            kernel_fn, verbose=True, random_state=self.random_state
        )

        self.trigger_model = self._dispatch_trigger_perturbation(
            kwargs["trigger_pert_type"],
            pert_num,
            pert_space,
            pert_val,
            embedding_module=embedding_module,
            init_pert_ind=torch.LongTensor(init_pert_ind) if init_pert_ind else None,
            trigger_perturbation_kwargs=kwargs,
        )

        self.register_buffer("lime_values", torch.zeros((pert_space,)))

    def _dispatch_trigger_perturbation(
        self,
        trigger_pert_type: str,
        pert_num: int,
        pert_space: int,
        pert_val: torch.Tensor,
        embedding_module: torch.nn.Module = None,
        init_pert_ind: torch.Tensor = None,
        trigger_perturbation_kwargs={},
    ) -> BaseTriggerPerturbation:
        if trigger_pert_type == "trigger_perturbation":
            return TriggerPerturbation(
                pert_num,
                pert_space,
                init_pert_val=pert_val,
                init_pert_ind=init_pert_ind,
            )
        elif trigger_pert_type == "embedding_trigger_perturbation":
            return EmbeddingTriggerPerturbation(
                pert_num,
                pert_space,
                embedding_module=embedding_module,
                init_pert_val=pert_val,
                init_pert_ind=init_pert_ind,
            )
        else:
            raise ValueError(f"Unknown trigger perturbation type {trigger_pert_type}")

    def _dispatch_mask_fn(self, mask_type: str, batch_size: int) -> MaskInput:
        mask_types: Dict[str, MaskInput] = {
            "random": RandomlyMaskInput,
        }
        mask_fn: MaskInput = mask_types[mask_type](self.random_seed, 1, batch_size)
        return mask_fn

    def state_dict(self) -> Dict[str, torch.Tensor]:
        state = {
            "trigger_model": self.trigger_model.state_dict(),
            "lime_values": self.lime_values,
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.trigger_model.load_state_dict(state_dict["trigger_model"])
        self.lime_values.copy_(state_dict["lime_values"])

    def perturb(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.trigger_model(x)

    def get_pert_ind(self, **kwargs) -> torch.Tensor:
        return self.trigger_model.get_buffer("pert_ind")

    def get_pert_val(self, **kwargs) -> torch.Tensor:
        return self.trigger_model.get_buffer("pert_val")

    @torch.inference_mode()
    def get_instance_importance(
        self,
        x: torch.Tensor,
        mask_fn: MaskInput,
        mask_value: torch.Tensor,
        forward_fn: Callable,
        label: int,
    ) -> torch.Tensor:
        """x: [1, feat_dim] or [1, C, H, W] or [1, T, feat_dim]"""
        list_of_data = []
        list_of_pred = []
        list_of_data.append(torch.ones_like(x).reshape(1, -1))
        list_of_pred.append(forward_fn(x))

        cur_neighbor_size = 1
        for masked_data, masked_samples in mask_fn.mask_sample(x, mask_value):
            max_neighbor_size = self.neighbor_size - cur_neighbor_size
            if max_neighbor_size == 0:
                break

            to_add_size = min(max_neighbor_size, self.masked_batch_size)
            list_of_pred.append(forward_fn(masked_samples[:to_add_size]))
            list_of_data.append(masked_data[:to_add_size].reshape(to_add_size, -1))
            cur_neighbor_size += to_add_size

        data: np.ndarray = torch.cat(list_of_data, dim=0).numpy(force=True)
        labels: np.ndarray = (
            torch.cat(list_of_pred, dim=0).unsqueeze(-1).numpy(force=True)
        )
        distances = sklearn.metrics.pairwise_distances(
            data, data[0].reshape(1, -1), metric=self.distance_metric
        ).ravel()

        _, used_features_and_score, *_ = self.lime_model.explain_instance_with_data(
            data,
            labels,
            distances,
            label,
            self.neighbor_feat_num,
            model_regressor=None,
            feature_selection=self.feature_selection,
        )
        score = torch.zeros_like(self.lime_values)
        for i, j in used_features_and_score:
            score[i] = j.item()
        return torch.abs(score)

    @torch.inference_mode()
    def get_global_importance(
        self,
        x: torch.Tensor,
        mask_fn: MaskInput,
        mask_value: torch.Tensor,
        forward_fn: Callable,
        label: int,
    ) -> torch.Tensor:
        """x: [N, feat_dim] or [N, C, H, W] or [N, T, feat_dim]"""

        xs = torch.split(x, 1, dim=0)
        scores = []
        for x in tqdm(xs):
            scores.append(
                self.get_instance_importance(x, mask_fn, mask_value, forward_fn, label)
            )

        scores = torch.stack(scores, dim=0)
        return torch.mean(scores, dim=0)

    def update(self, x: torch.Tensor, y: torch.Tensor = None, **kwargs) -> None:
        mask_fn = self._dispatch_mask_fn(self.mask_type, self.masked_batch_size)
        global_importance = self.get_global_importance(
            x,
            mask_fn=mask_fn,
            mask_value=kwargs["mask_value"],
            forward_fn=kwargs["forward_fn"],
            label=kwargs["label"],
        )

        self.lime_values.copy_(global_importance)
        self.trigger_model.pert_ind.copy_(
            global_importance.argsort(descending=True)[: self.pert_num]
        )
