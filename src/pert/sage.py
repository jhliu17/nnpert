import numpy as np
import torch
import torch.nn as nn
import sage

from sage.imputers import MarginalImputer, DefaultImputer
from math import ceil
from typing import Callable, Dict, List
from dataclasses import dataclass, field
from .base import PertModule
from .nn.module import (
    BaseTriggerPerturbation,
    TriggerPerturbation,
    EmbeddingTriggerPerturbation,
)


@dataclass
class SAGEPertModuleConfig:
    n_jobs: int = 1

    batch_size: int = 128

    n_permutations: int = 1024

    detect_convergence: bool = False

    random_state: int = 2024

    trial_size: int = 4

    # trigger pert type
    trigger_pert_type: str = "trigger_perturbation"


class TrialMarginalImputer(MarginalImputer):
    def __init__(self, model, data, trial_size: int):
        super().__init__(model, data)
        self.trial_size = trial_size

    def __call__(self, x, S):
        xs = np.array_split(x, ceil(len(x) / self.trial_size))
        Ss = np.array_split(S, ceil(len(x) / self.trial_size))
        preds = []
        for x_, S_ in zip(xs, Ss):
            pred = super().__call__(x_, S_)
            preds.append(pred)
        return np.concatenate(preds)


class SAGEPertModule(PertModule):
    def __init__(
        self,
        pert_num: int,
        pert_space: int,
        pert_val: torch.Tensor,
        embedding_module: torch.nn.Module = None,
        n_jobs: int = 1,
        batch_size: int = 128,
        trial_size: int = 4,
        n_permutations: int = 1024,
        detect_convergence: bool = False,
        random_state: int = 2024,
        init_pert_ind: List[int] = [],
        **kwargs,
    ) -> None:
        super().__init__()
        self.pert_val = pert_val
        self.pert_num = pert_num
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.trial_size = trial_size
        self.n_permutations = n_permutations
        self.random_state = random_state
        self.detect_convergence = detect_convergence

        self.trigger_model = self._dispatch_trigger_perturbation(
            kwargs["trigger_pert_type"],
            pert_num,
            pert_space,
            pert_val,
            embedding_module=embedding_module,
            init_pert_ind=torch.LongTensor(init_pert_ind) if init_pert_ind else None,
            trigger_perturbation_kwargs=kwargs,
        )
        self.register_buffer("sage_values", torch.zeros((pert_space,)))

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

    def state_dict(self) -> Dict[str, torch.Tensor]:
        state = {
            "trigger_model": self.trigger_model.state_dict(),
            "sage_values": self.sage_values,
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.trigger_model.load_state_dict(state_dict["trigger_model"])
        self.sage_values.copy_(state_dict["sage_values"])

    def perturb(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.trigger_model(x)

    def get_pert_ind(self, **kwargs) -> torch.Tensor:
        return self.trigger_model.get_buffer("pert_ind")

    def get_eval_fn(self, forward_fn: Callable, device: torch.device):
        @torch.inference_mode()
        def eval_fn(x: np.ndarray) -> np.ndarray:
            y = forward_fn(torch.tensor(x, dtype=torch.float32, device=device)).numpy(
                force=True
            )
            return y

        return eval_fn

    def get_loss_fn(self, head_loss_fn: Callable, device: torch.device):
        @torch.inference_mode()
        def loss_fn(y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
            y_pred = torch.from_numpy(y_pred).to(device)
            y = torch.from_numpy(y).to(device)
            return head_loss_fn(y_pred, y).numpy(force=True)

        return loss_fn

    def update(self, x: torch.Tensor, y: torch.Tensor = None, **kwargs) -> None:
        # imputer = TrialMarginalImputer(
        #     self.get_eval_fn(kwargs["forward_fn"], kwargs["device"]),
        #     x.numpy(force=True),
        #     self.trial_size,
        # )
        imputer = DefaultImputer(
            self.get_eval_fn(kwargs["forward_fn"], kwargs["device"]),
            self.pert_val.numpy(force=True),
        )
        estimator = sage.PermutationEstimator(
            imputer, loss="mse", n_jobs=self.n_jobs, random_state=self.random_state
        )

        estimator.loss_fn = self.get_loss_fn(kwargs["head_loss_fn"], kwargs["device"])
        explanation = estimator(
            x.numpy(force=True),
            y.numpy(force=True),
            detect_convergence=self.detect_convergence,
            batch_size=self.batch_size,
            n_permutations=self.n_permutations if self.n_permutations > 0 else None,
            min_coalition=max(x.size(-1) - self.pert_num, 0),
            max_coalition=x.size(-1),
            verbose=True,
        )

        # update buffer
        self.sage_values.copy_(torch.tensor(explanation.values, dtype=torch.float32))
        self.trigger_model.pert_ind.copy_(
            torch.tensor(explanation.values).argsort(descending=True)[: self.pert_num]
        )
