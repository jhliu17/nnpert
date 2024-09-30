import torch
import torch.nn as nn
import numpy as np

from typing import Callable, Dict, List, Literal
from dataclasses import dataclass, field
from ..base import PertModule
from ..nn.module import (
    BaseTriggerPerturbation,
    TriggerPerturbation,
    EmbeddingTriggerPerturbation,
)
from .train import create_dataset, CXPlainTrainerConfig, train_cxplain
from ..utils.mask import MaskInputImage, MaskInputSequence, MaskInput


@dataclass
class CXPlainPertModuleConfig:
    mask_type: Literal["image", "sequence"] = "image"

    batch_size: int = 128

    masked_batch_size: int = 128

    # trigger pert type
    trigger_pert_type: str = "trigger_perturbation"

    trainer_config: CXPlainTrainerConfig = field(default_factory=CXPlainTrainerConfig)


class CXPlainPertModule(PertModule):
    def __init__(
        self,
        pert_num: int,
        pert_space: int,
        pert_val: torch.Tensor,
        embedding_module: torch.nn.Module = None,
        mask_type: str = "image",
        trainer_config: dict = None,
        batch_size: int = 128,
        masked_batch_size: int = 128,
        init_pert_ind: List[int] = [],
        **kwargs,
    ) -> None:
        super().__init__()
        self.pert_val = pert_val
        self.pert_num = pert_num
        self.mask_type = mask_type
        self.trainer_config = CXPlainTrainerConfig(**trainer_config)
        self.batch_size = batch_size
        self.masked_batch_size = masked_batch_size

        self.trigger_model = self._dispatch_trigger_perturbation(
            kwargs["trigger_pert_type"],
            pert_num,
            pert_space,
            pert_val,
            embedding_module=embedding_module,
            init_pert_ind=torch.LongTensor(init_pert_ind) if init_pert_ind else None,
            trigger_perturbation_kwargs=kwargs,
        )
        self.explain_model: nn.Module = nn.Identity()
        self.register_buffer("cxplain_values", torch.zeros((pert_space,)))

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
            "image": MaskInputImage,
            "sequence": MaskInputSequence,
        }
        mask_fn: MaskInput = mask_types[mask_type](
            self.trainer_config.downsample_factors, batch_size
        )
        return mask_fn

    def state_dict(self) -> Dict[str, torch.Tensor]:
        state = {
            "trigger_model": self.trigger_model.state_dict(),
            "explain_model": self.explain_model.state_dict(),
            "cxplain_values": self.cxplain_values,
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.trigger_model.load_state_dict(state_dict["trigger_model"])
        # self.explain_model.load_state_dict(state_dict["explain_model"])
        self.cxplain_values.copy_(state_dict["cxplain_values"])

    def perturb(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.trigger_model(x)

    def get_pert_ind(self, **kwargs) -> torch.Tensor:
        return self.trigger_model.get_buffer("pert_ind")

    def get_pert_val(self, **kwargs) -> torch.Tensor:
        return self.trigger_model.get_buffer("pert_val")

    @torch.inference_mode()
    def get_instance_importance(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, feat_dim] or [N, C, H, W] or [N, T, feat_dim]"""
        self.explain_model.eval()
        score = self.explain_model(x).exp()  # [N, feat_dim] or [N, H*W] or [N, T]
        return score

    @torch.inference_mode()
    def get_global_importance(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, feat_dim] or [N, C, H, W] or [N, T, feat_dim]"""
        self.explain_model.eval()
        xs = torch.split(x, self.batch_size, dim=0)
        scores = []
        for x in xs:
            scores.append(self.get_instance_importance(x))

        scores = torch.cat(scores, dim=0)
        return torch.mean(scores, dim=0)

    def update(self, x: torch.Tensor, y: torch.Tensor = None, **kwargs) -> None:
        mask_fn = self._dispatch_mask_fn(self.mask_type, self.masked_batch_size)
        train_ds = create_dataset(
            x,
            forward_fn=kwargs["forward_fn"],
            loss_fn=kwargs["head_loss_fn"],
            mask_fn=mask_fn,
            mask_value=kwargs["mask_value"],
        )
        eval_ds = create_dataset(
            kwargs["x_eval"],
            forward_fn=kwargs["forward_fn"],
            loss_fn=kwargs["head_loss_fn"],
            mask_fn=mask_fn,
            mask_value=kwargs["mask_value"],
        )

        self.explain_model = train_cxplain(
            train_ds,
            eval_ds,
            kwargs["device"],
            self.trainer_config,
        )

        global_importance = mask_fn.assign_important_score(
            x[0:1], self.get_global_importance(x)
        ).flatten()
        self.cxplain_values.copy_(global_importance)
        self.trigger_model.pert_ind.copy_(
            global_importance.argsort(descending=True)[: self.pert_num]
        )
