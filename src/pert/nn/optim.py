import torch
import torch.nn as nn

from torch.optim import Optimizer
from typing import Callable, List, Union
from .op import create_trans_pert_vecs_from_ind


class SingleTriggerReplacementStrategy(Optimizer):
    def __init__(
        self,
        param: torch.Tensor,
        pert_model_param: nn.Parameter,
        update_strategy: str = "greedy",
        num_candidates: int = 1,
        invalid_inds: Union[List[int], None] = None,
    ) -> None:
        if not isinstance(param, torch.Tensor):
            raise ValueError(
                "For `strategy`, it can take only one perturbation tensor to optimize."
            )
        if param.dim() != 1:
            raise ValueError("For perturbation tensor, its dimension should be one.")

        params = [{"params": param}, {"params": pert_model_param}]
        defaults = {
            "update_strategy": update_strategy,
            "invalid_inds": invalid_inds,
            "num_candidates": num_candidates,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, eval_function: Callable = None):
        """Perform single step optimization, the optimizer always assumes the objective
        function is needed to be minimized.

        :param eval_function: eval function for each candidate, defaults to None
        """
        # group 1 is the transition matrix
        trans_group = self.param_groups[1]
        p: torch.Tensor
        p = trans_group["params"][0]
        if p.grad is None:
            raise Exception(f"The transition matrix of {p} is empty.")

        trans = -p.grad
        invalid_inds, num_candidates = (
            trans_group["invalid_inds"],
            trans_group["num_candidates"],
        )

        # mask out invalid inds if existed
        if invalid_inds is not None:
            invalid_index = torch.tensor(invalid_inds)
            trans.index_fill_(-1, invalid_index, float("-inf"))

        best_k_inds_score, best_k_inds = torch.topk(trans, num_candidates, dim=-1)

        # update perturbation tensor in group 0
        update_group = self.param_groups[0]
        pert_ind: torch.Tensor = update_group["params"][0]

        row: int
        col: int
        if trans_group["update_strategy"] == "coordinate":
            state = self.state[pert_ind]

            # State initialization
            if len(state) == 0:
                state["coordinate"] = 0

            coordinate = state["coordinate"] % trans.shape[0]
            state["coordinate"] = coordinate + 1

            row = coordinate
            if eval_function is not None:
                pert_vec_cands = create_trans_pert_vecs_from_ind(
                    pert_ind, best_k_inds, trans.shape[1]
                )
                loss_per_candidate = eval_function(
                    pert_vec_cands[row * num_candidates : (row + 1) * num_candidates]
                )

                # minimize the loss
                col = torch.argmin(loss_per_candidate).item()
            else:
                col = torch.argmax(best_k_inds_score[row]).item()
        else:
            # greedy selection is adopted
            if eval_function is not None:
                pert_vec_cands = create_trans_pert_vecs_from_ind(pert_ind, best_k_inds)
                loss_per_candidate = eval_function(pert_vec_cands)

                # minimize the loss
                min_ind = torch.argmin(loss_per_candidate)
                row, col = divmod(min_ind.item(), num_candidates)
            else:
                max_ind = torch.argmax(best_k_inds_score)
                row, col = divmod(max_ind.item(), num_candidates)

        # update perturbation tensor in group 0
        pert_ind[row] = best_k_inds[row, col]
