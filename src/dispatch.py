import torch

from .pert import build_pert_module, PertModule, PertModuleConfig


def dispatch_pert_module(
    pert_type: str,
    pert_num: int,
    pert_space: int,
    pert_val: torch.Tensor,
    pert_step: int,
    pert_config: PertModuleConfig,
    embedding_module: torch.nn.Module = None,
) -> PertModule:
    """Build according perturbation module based on pert_type and pert_config.

    :param pert_type: pert module type
    :param pert_num: num of perturbations
    :param pert_space: perturbation space
    :param pert_val: perturbation value
    :param pert_config: general pert config
    :param embedding_module: possible embedding module
    :return: created perturbation module
    """

    return build_pert_module(
        pert_type=pert_type,
        pert_num=pert_num,
        pert_space=pert_space,
        pert_val=pert_val,
        pert_step=pert_step,
        pert_config=pert_config,
        embedding_module=embedding_module,
    )
