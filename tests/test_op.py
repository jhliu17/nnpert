import torch
import sys
import os
import pytest
from pathlib import Path

src_folder = f"{Path(os.path.abspath(__file__)).parent.parent}"
sys.path.insert(0, src_folder)
from src.pert.nn.op import (  # noqa: E402
    create_pert_vec,
    create_trans_pert_vecs,
    create_trans_pert_vecs_from_ind,
    SequenceTriggerPerturbationFunction,
)


def test_create_pert_vec():
    pert_vec = create_pert_vec(torch.tensor([1, 2, 4, 5]), 6)
    tagt_vec = torch.tensor([0, 1, 1, 0, 1, 1]).float()

    assert torch.allclose(pert_vec, tagt_vec)


def test_create_trans_pert_vecs():
    pert_vec = create_pert_vec(torch.tensor([1, 5]), 6)
    trans_inds = torch.tensor([[0, 5, 1], [3, 1, 2]])
    trans_vec = create_trans_pert_vecs(pert_vec, trans_inds)

    # should correctly generate a batch of perturbation vector given a transition matrix
    expect_trans_vec = torch.tensor(
        [
            [1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
        ]
    ).float()
    assert torch.allclose(trans_vec, expect_trans_vec)

    # should raise value errr if perturbation number mismatch with transition state
    with pytest.raises(ValueError):
        pert_vec = create_pert_vec(torch.tensor([1, 5]), 6)
        trans_inds = torch.tensor([[0, 5, 1]])
        trans_vec = create_trans_pert_vecs(pert_vec, trans_inds)


def test_create_trans_pert_vecs_from_ind():
    pert_ind = torch.tensor([5, 1])
    trans_inds = torch.tensor([[3, 1, 2], [0, 5, 1]])
    trans_vec = create_trans_pert_vecs_from_ind(pert_ind, trans_inds, 6)

    # should correctly generate a batch of perturbation vector given a transition matrix
    expect_trans_vec = torch.tensor(
        [
            [0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1],
        ]
    ).float()
    assert torch.allclose(trans_vec, expect_trans_vec)


def test_trigger_perturbation_function():
    x = torch.randn((2, 6))
    pert_ind = torch.tensor([1, 5])
    pert_val = torch.randn((6,))

    # test forward is ok
    pert_x = SequenceTriggerPerturbationFunction.apply(
        x, pert_ind, pert_val, torch.empty((2, 6))
    )
    expect_pert_x = torch.tensor(
        [
            [x[0][0], pert_val[1], x[0][2], x[0][3], x[0][4], pert_val[5]],
            [x[1][0], pert_val[1], x[1][2], x[1][3], x[1][4], pert_val[5]],
        ]
    )
    assert torch.allclose(pert_x, expect_pert_x)

    # test backward gradient is ok
    grad_output = torch.rand_like(pert_x)
    grad_placeholder = torch.empty((2, 6), requires_grad=True)
    pert_x = SequenceTriggerPerturbationFunction.apply(
        x, pert_ind, pert_val, grad_placeholder
    )
    pert_x.backward(gradient=grad_output)
    assert grad_placeholder.grad.shape == torch.Size([2, 6])

    diff_pert = pert_val - x
    trans_matrix = grad_placeholder.grad
    assert torch.allclose(trans_matrix[0, 1], torch.tensor(0.0))
    assert torch.allclose(trans_matrix[1, 5], torch.tensor(0.0))
    assert torch.allclose(
        trans_matrix[0, 0],
        torch.mean(
            torch.tensor(
                [
                    diff_pert[0, 0] * grad_output[0, 0]
                    - diff_pert[0, 1] * grad_output[0, 1],
                    diff_pert[1, 0] * grad_output[1, 0]
                    - diff_pert[1, 1] * grad_output[1, 1],
                ]
            )
        ),
    )
    assert torch.allclose(
        trans_matrix[1, 3],
        torch.mean(
            torch.tensor(
                [
                    diff_pert[0, 3] * grad_output[0, 3]
                    - diff_pert[0, 5] * grad_output[0, 5],
                    diff_pert[1, 3] * grad_output[1, 3]
                    - diff_pert[1, 5] * grad_output[1, 5],
                ]
            )
        ),
    )
    assert torch.allclose(
        trans_matrix[0, 5],
        torch.mean(
            torch.tensor(
                [
                    -diff_pert[0, 1] * grad_output[0, 1],
                    -diff_pert[1, 1] * grad_output[1, 1],
                ]
            )
        ),
    )

    # check approximation
    expected_trans_matrix = torch.empty_like(trans_matrix)
    with torch.inference_mode():
        pert_x = SequenceTriggerPerturbationFunction.apply(
            x, pert_ind, pert_val, torch.empty((2, 6))
        )
        for i in range(len(pert_ind)):
            for j in range(6):
                updated_pert_ind = pert_ind.detach().clone()
                updated_pert_ind[i] = j

                up_pert_x = SequenceTriggerPerturbationFunction.apply(
                    x, updated_pert_ind, pert_val, torch.empty((2, 6))
                )

                expected_trans_matrix[i, j] = torch.mean(
                    torch.sum((up_pert_x - pert_x) * grad_output, dim=-1)
                )

    print(expected_trans_matrix)
    assert torch.allclose(trans_matrix, expected_trans_matrix)
