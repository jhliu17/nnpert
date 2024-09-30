import torch

from typing import Literal
from dataclasses import dataclass, field, asdict
from .base import PertModule
from .trigger import TriggerPertModule, TriggerPertModuleConfig
from .sage import SAGEPertModule, SAGEPertModuleConfig
from .cxplain import CXPlainPertModule, CXPlainPertModuleConfig
from .lime import LIMEPertModule, LIMEPertModuleConfig


pert_dict = {
    "trigger": TriggerPertModule,
    "sage": SAGEPertModule,
    "cxplain": CXPlainPertModule,
    "lime": LIMEPertModule,
}


@dataclass
class PertModuleConfig:
    # pert model name
    model_type: Literal["trigger", "sage", "cxplain", "lime"] = (
        "trigger"
    )

    # perturbation num
    perturbation_num: int = 8

    # trigger perturbation configurations
    trigger: TriggerPertModuleConfig = field(default_factory=TriggerPertModuleConfig)

    # trigger with sage perturbation configurations
    sage: SAGEPertModuleConfig = field(default_factory=SAGEPertModuleConfig)

    # trigger with cxplain perturbation configurations
    cxplain: CXPlainPertModuleConfig = field(default_factory=CXPlainPertModuleConfig)

    lime: LIMEPertModuleConfig = field(default_factory=LIMEPertModuleConfig)


def build_pert_module(
    pert_type: str,
    pert_num: int,
    pert_space: int,
    pert_val: torch.Tensor,
    pert_step: int,
    pert_config: PertModuleConfig,
    embedding_module: torch.nn.Module = None,
) -> PertModule:
    return pert_dict[pert_type](
        pert_num=pert_num,
        pert_space=pert_space,
        pert_val=pert_val,
        pert_step=pert_step,
        embedding_module=embedding_module,
        **asdict(getattr(pert_config, pert_type)),
    )
