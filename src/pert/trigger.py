import torch

from typing import Callable, Dict, List
from dataclasses import dataclass, field
from .base import PertModule
from .nn.module import (
    BaseTriggerPerturbation,
    TriggerPerturbation,
    EmbeddingTriggerPerturbation,
)
from .nn.optim import SingleTriggerReplacementStrategy
from ..utils import group_mean


@dataclass
class TriggerPertModuleConfig:
    # replacement strategy num
    replace_num_candidates: int = 8

    # trigger pert type
    trigger_pert_type: str = "trigger_perturbation"

    # use optim eval
    use_optim_eval: bool = True

    # init perturbation index
    init_pert_ind: List[int] = field(default_factory=list)


class TriggerPertModule(PertModule):
    def __init__(
        self,
        pert_num: int,
        pert_space: int,
        pert_val: torch.Tensor,
        replace_num_candidates: int,
        embedding_module: torch.nn.Module = None,
        use_optim_eval: bool = False,
        init_pert_ind: List[int] = [],
        **kwargs,
    ) -> None:
        super().__init__()
        self.update_strategy = "coordinate"
        self.replace_num_candidates = replace_num_candidates
        self.use_optim_eval = use_optim_eval
        self.embedding_module = embedding_module

        self.trigger_model = self._dispatch_trigger_perturbation(
            kwargs["trigger_pert_type"],
            pert_num,
            pert_space,
            pert_val,
            embedding_module=embedding_module,
            init_pert_ind=torch.LongTensor(init_pert_ind) if init_pert_ind else None,
            trigger_perturbation_kwargs=kwargs,
        )

        self.trigger_opt = self._create_optim(
            self.trigger_model, self.update_strategy, self.replace_num_candidates
        )

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

    @staticmethod
    def _create_optim(
        trigger_model: TriggerPerturbation,
        update_strategy: str,
        replace_num_candidates: int,
    ):
        opt = SingleTriggerReplacementStrategy(
            trigger_model.get_buffer("pert_ind"),
            trigger_model.get_parameter("weight"),
            update_strategy=update_strategy,
            num_candidates=replace_num_candidates,
        )
        return opt

    def to(self, device: torch.device):
        super().to(device)

        # correctly create optimizer after moving the module to device
        self.trigger_opt = self._create_optim(
            self.trigger_model, self.update_strategy, self.replace_num_candidates
        )
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        state = {
            "trigger_model": self.trigger_model.state_dict(),
            "trigger_opt": self.trigger_opt.state_dict(),
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.trigger_model.load_state_dict(state_dict["trigger_model"])
        self.trigger_opt.load_state_dict(state_dict["trigger_opt"])

    def perturb(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.trigger_model(x)

    def get_pert_ind(self, **kwargs) -> torch.Tensor:
        return self.trigger_model.get_buffer("pert_ind")

    def get_optim_eval_function(
        self,
        inp: torch.Tensor,
        forward_fn: Callable,
        eval_fn: Callable,
        eval_trial_size: int,
    ):
        @torch.inference_mode()
        def eval_function(pert_vecs: torch.Tensor):
            batch_size = inp.shape[0]
            trial_num = pert_vecs.shape[0]
            trial_iter = torch.split(
                pert_vecs, eval_trial_size if eval_trial_size > 0 else trial_num, dim=0
            )

            sub_group_loss = []
            for sub_pert_vecs in trial_iter:
                sub_trial_num = sub_pert_vecs.shape[0]
                pert_inp = self.trigger_model.trial(inp, sub_pert_vecs)
                pred_traj = forward_fn(pert_inp)
                loss = eval_fn(pred_traj)
                group_loss = group_mean(loss, sub_trial_num, batch_size)
                sub_group_loss.append(group_loss.flatten())
            return torch.cat(sub_group_loss, dim=0)

        return eval_function

    def update(self, x: torch.Tensor, y: torch.Tensor = None, **kwargs) -> None:
        if self.use_optim_eval:
            self.trigger_opt.step(
                self.get_optim_eval_function(
                    x,
                    kwargs["forward_fn"],
                    kwargs["per_sample_eval_fn"],
                    kwargs.get("eval_trial_size", -1),
                )
            )
        else:
            self.trigger_opt.step()
        self.trigger_opt.zero_grad()
