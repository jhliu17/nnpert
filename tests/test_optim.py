import torch
import sys
import os

from pathlib import Path

src_folder = f"{Path(os.path.abspath(__file__)).parent.parent}"
sys.path.insert(0, src_folder)
from src.pert.nn.optim import SingleTriggerReplacementStrategy  # noqa: E402
from src.pert.nn.op import SequenceTriggerPerturbationFunction  # noqa: E402


def test_single_trigger_replacement_strategy():
    torch.random.manual_seed(2023)
    x = torch.randn((3, 6))
    pert_ind = torch.tensor([1, 5])
    pert_val = torch.randn((6,))

    # test backward gradient is ok
    grad_output = torch.randn((3, 6))
    grad_placeholder = torch.empty((2, 6), requires_grad=True)
    replace_optim = SingleTriggerReplacementStrategy(
        pert_ind, grad_placeholder, num_candidates=3
    )

    pert_x = SequenceTriggerPerturbationFunction.apply(
        x, pert_ind, pert_val, grad_placeholder
    )

    assert torch.allclose(pert_ind, torch.tensor([1, 5]))
    replace_optim.zero_grad()
    print(grad_placeholder.grad)
    pert_x.backward(gradient=grad_output)
    print(grad_placeholder.grad)
    replace_optim.step()
    except_updated_pert_ind = torch.tensor([2, 5])
    assert torch.allclose(pert_ind, except_updated_pert_ind)

    replace_optim.zero_grad()
    print(grad_placeholder.grad)
