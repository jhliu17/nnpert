import torch
import sys
import os

from pathlib import Path

src_folder = f"{Path(os.path.abspath(__file__)).parent.parent}"
sys.path.insert(0, src_folder)
from src.pert import PertModuleConfig, TriggerPertModuleConfig  # noqa: E402
from src.dispatch import dispatch_pert_module  # noqa: E402


def test_pert_module():
    torch.random.manual_seed(2023)
    x = torch.randn((3, 6))
    pert_ind = [1, 5]
    pert_val = torch.randn((6,))

    # test backward gradient is ok
    grad_output = torch.randn((3, 6))
    pert_module = dispatch_pert_module(
        "trigger",
        2,
        6,
        pert_val,
        pert_step=1,
        pert_config=PertModuleConfig(
            2,
            trigger=TriggerPertModuleConfig(
                replace_num_candidates=3, use_optim_eval=False, init_pert_ind=pert_ind
            ),
        ),
    )

    pert_x = pert_module.perturb(x)

    assert torch.allclose(pert_module.trigger_model.pert_ind, torch.tensor([1, 5]))
    pert_module.trigger_opt.zero_grad()
    assert pert_module.trigger_model.weight.grad is None
    pert_x.backward(gradient=grad_output)
    pert_module.update(x)
    except_updated_pert_ind = torch.tensor([2, 5])
    assert torch.allclose(pert_module.trigger_model.pert_ind, except_updated_pert_ind)

    pert_module.trigger_opt.zero_grad()
    assert pert_module.trigger_model.weight.grad is None


@torch.inference_mode()
def test_pert_module_inference():
    torch.random.manual_seed(2023)
    x = torch.randn((3, 6))
    pert_ind = [1, 5]
    pert_val = torch.randn((6,))

    # test backward gradient is ok
    grad_output = torch.randn((3, 6))
    pert_module = dispatch_pert_module(
        "trigger",
        2,
        6,
        pert_val,
        pert_step=1,
        pert_config=PertModuleConfig(
            2,
            trigger=TriggerPertModuleConfig(
                replace_num_candidates=3, use_optim_eval=False, init_pert_ind=pert_ind
            ),
        ),
    )

    pert_x = pert_module.perturb(x)

    assert torch.allclose(pert_module.trigger_model.pert_ind, torch.tensor([1, 5]))
    assert pert_module.trigger_model.weight.grad is None
