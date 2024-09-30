import torch
import torch.nn as nn
import wandb
import numpy as np

from tqdm import tqdm
from typing import Callable, List, Tuple
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import v2
from einops import rearrange
from dataclasses import dataclass, asdict, field

from .eval import eval_cell_type_ratio
from .utils import (
    divisible_by,
    sample_vector_field,
    visualize_trajs,
    visualize_vector_field,
    cycle_dataloader,
    read_and_post_process_adata,
    visualize_changed_mnist_digit,
)
from .trajectory.sample import sample_trajectory, sample_deep_velo_trajectory_from_ckpt
from .head.base import HeadModule
from .dataset import GeneTrajDataset, MNISTSubset
from .pert import PertModule
from .train.base import BaseTrainer


@dataclass
class MNISTPerturbationTrainerConfig:
    # data folder
    data_folder: str

    # output folder
    output_folder: str

    # perturbation total step
    pert_num_steps: int = 100000

    # perturbation state step
    pert_state_step: int = -1

    # save and sample every
    save_and_sample_every: int = 1000

    # starting num type
    starting_num_type: str = "8"

    train_set_ratio: float = 0.8

    # model training batch size
    train_batch_size: int = 64

    # model evaluation batch size
    eval_batch_size: int = 64

    # cuda
    use_cuda: bool = True

    num_workers: int = 0

    # visualize sample numbers
    num_samples: int = 8

    # random seed
    seed: int = 2024


class MNISTPerturbationTrainer(BaseTrainer):
    def __init__(
        self,
        pert_model: PertModule,
        model: nn.Module,
        head_model: HeadModule,
        num_type_dict: dict[str, int],
        *,
        args: MNISTPerturbationTrainerConfig,
        has_wandb_writer: bool = False,
    ) -> None:
        super().__init__(args.output_folder, has_wandb_writer)
        self.use_cuda = args.use_cuda
        self.seed = args.seed
        self.save_and_sample_every = args.save_and_sample_every
        self.starting_num_type = args.starting_num_type
        self.num_type_dict = num_type_dict
        self.ind2num_type_dict = {v: k for k, v in num_type_dict.items()}
        self.num_samples = args.num_samples

        self.model = model.to(self.device)
        self.pert_model = pert_model.to(self.device)
        self.head_model = head_model.to(self.device)

        # dummy optimizer for clean gradient on head model
        self.head_model_opt = SGD(self.head_model.parameters(), lr=1e-3)

        # step counter state
        self.pert_step = 0
        self.pert_num_steps = args.pert_num_steps

        train_dataset, eval_dataset, test_dataset = self.build_datasets(
            args.data_folder, args.train_set_ratio
        )

        dataloader_worker = args.num_workers
        self.dl = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=dataloader_worker,
        )

        self.eval_dl = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=dataloader_worker,
        )
        self.test_dl = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=dataloader_worker,
        )

    @property
    def device(self):
        return torch.device("cuda") if self.use_cuda else torch.device("cpu")

    def build_datasets(self, root, train_set_ratio):
        mnist_dataset = MNIST(root, train=False, download=True)

        # create subset dataset from the whole mnist dataset from starting_num_type
        subset_index = []
        for i in range(len(mnist_dataset)):
            if str(mnist_dataset[i][1]) == self.starting_num_type:
                subset_index.append(i)
        subset_dataset = MNISTSubset(
            mnist_dataset,
            subset_index,
            self.num_type_dict,
            transform=v2.Compose(
                [
                    v2.PILToTensor(),
                    v2.Lambda(lambda x: x / 255),
                    v2.Lambda(lambda x: x.flatten()),
                ]
            ),
        )

        generator = torch.Generator().manual_seed(self.seed)
        train_size = int(train_set_ratio * len(subset_dataset))
        test_size = len(subset_dataset) - train_size
        train_dataset, test_dataset = random_split(
            subset_dataset, [train_size, test_size], generator
        )

        train_size = int(train_set_ratio * len(train_dataset))
        eval_size = len(train_dataset) - train_size
        train_dataset, eval_dataset = random_split(
            train_dataset, [train_size, eval_size], generator
        )

        return train_dataset, eval_dataset, test_dataset

    def get_state(self):
        state = {
            "pert_step": self.pert_step,
            "pert_model": self.pert_model.state_dict(),
            "head_model": self.head_model.state_dict(),
        }
        return state

    def load_state(self, state):
        self.pert_step = state["pert_step"]
        self.pert_model.load_state_dict(state["pert_model"])
        self.head_model.load_state_dict(state["head_model"])

    @property
    def global_step(self) -> int:
        return self.pert_step

    def set_model_state(self, train: bool = True):
        """In this trainer, we only train on the perturbation model.

        :param train: whether to train mode, defaults to True
        """
        self.model.eval()
        self.head_model.eval()
        if train:
            self.pert_model.train()
        else:
            self.pert_model.eval()

    @torch.inference_mode()
    def generate_prediction(
        self, dataloader: DataLoader, pert_model: PertModule = None
    ):
        inputs, prediction = [], []
        for inp, tgt in dataloader:
            inp = inp.to(self.device)
            tgt = tgt.to(self.device)

            if pert_model is not None:
                inp = pert_model.perturb(inp)

            logit = self.head_model.classifier(inp)

            inputs.append(inp.numpy(force=True))
            prediction.append(logit.argmax(dim=1).numpy(force=True))
        return np.concatenate(inputs), np.concatenate(prediction)

    @torch.inference_mode()
    def eval(self, dataloader: DataLoader):
        # turn model to eval mode
        self.set_model_state(train=False)

        log_results = {}

        # class ratio, visualize perturbation
        inputs, preds = self.generate_prediction(dataloader)
        inputs_pert, preds_pert = self.generate_prediction(dataloader, self.pert_model)

        for num, ind in self.num_type_dict.items():
            log_results[f"class_ratio/{num}"] = (preds == ind).mean()
            log_results[f"class_ratio_pert/{num}"] = (preds_pert == ind).mean()

        # plot changed images
        changed_mask = np.logical_and(
            preds != preds_pert, preds == self.num_type_dict[self.starting_num_type]
        )
        if changed_mask.sum() == 0:
            changed_mask = np.array([True] * len(preds))
        fig = visualize_changed_mnist_digit(
            inputs[changed_mask][: self.num_samples],
            preds[changed_mask][: self.num_samples],
            inputs_pert[changed_mask][: self.num_samples],
            preds_pert[changed_mask][: self.num_samples],
            self.ind2num_type_dict,
        )
        log_results["visualization"] = wandb.Image(fig)

        return log_results

    @torch.inference_mode()
    def eval_during_training(self):
        results = self.eval(self.eval_dl)
        self.set_model_state(train=True)
        return results

    def get_forward_fn(self):
        def forward_fn(inp: torch.Tensor) -> torch.Tensor:
            logit = self.model(inp)
            return logit

        return forward_fn

    def get_per_sample_eval_fn(self):
        @torch.inference_mode()
        def eval_fn(final_state: torch.Tensor) -> torch.Tensor:
            loss = self.head_model.get_loss(final_state, reduction="none")
            return loss

        return eval_fn

    def train(self):
        self.set_model_state(train=True)

        # eval before perturbation
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")

        # start perturbating
        dl = cycle_dataloader(self.dl)
        with tqdm(
            initial=self.pert_step,
            total=self.pert_num_steps,
        ) as pbar:
            while self.pert_step < self.pert_num_steps:
                inp, tgt = next(dl)
                inp = inp.to(self.device)
                tgt = tgt.to(self.device)

                pert_inp = self.pert_model.perturb(
                    inp,
                    step=self.pert_step,
                    update_tau=True,
                )
                final_state = self.get_forward_fn()(pert_inp)
                loss = self.head_model.get_loss(final_state)
                loss.backward()

                pbar.set_description(f"loss: {loss.item():.4f}")
                self.log({"loss": loss.item()}, section="pert")

                # update perturbation optimizaer, and clean up model's gradient
                self.pert_model.update(
                    inp,
                    forward_fn=self.get_forward_fn(),
                    per_sample_eval_fn=self.get_per_sample_eval_fn(),
                )
                self.head_model.zero_grad()

                self.pert_step += 1
                if self.pert_step != 0 and divisible_by(
                    self.pert_step, self.save_and_sample_every
                ):
                    log_results = self.eval_during_training()
                    self.log(log_results, section="eval")
                    milestone = self.pert_step // self.save_and_sample_every
                    self.save(milestone)

                pbar.update(1)

        self.save("final")
        log_results = self.eval(self.test_dl)
        self.log(log_results, section="test")
        print("perturbation complete")


class MNISTSAGEPerturbationTrainer(MNISTPerturbationTrainer):
    def train(self):
        self.set_model_state(train=True)

        # eval before perturbation
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")

        # start perturbating
        xs = []
        ys = []
        for data, *_ in self.dl:
            data = data.to(self.device)
            xs.append(data)
            ys.append(self.get_forward_fn()(data))
        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)

        def get_forward_fn() -> Callable:
            self.model.eval()
            return self.get_forward_fn()

        def head_loss_fn(x: torch.Tensor, y: torch.Tensor):
            self.head_model.eval()
            return self.head_model.get_loss(x)

        # update perturbation optimizaer, and clean up model's gradient
        self.set_model_state(train=False)
        self.pert_model.update(
            x=x,
            y=y,
            forward_fn=get_forward_fn(),
            device=self.device,
            head_loss_fn=head_loss_fn,
        )

        self.pert_step += 1
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")
        log_results = self.eval(self.test_dl)
        self.log(log_results, section="test")

        self.save("final")
        print("perturbation complete")


class MNISTCXPlainPerturbationTrainer(MNISTPerturbationTrainer):
    def _get_pert_model_dataset(self, dl: DataLoader) -> torch.Tensor:
        xs = []
        for data, *_ in dl:
            data = data.reshape(-1, 28, 28).to(self.device)
            xs.append(data.unsqueeze(1))
        x = torch.cat(xs, dim=0)
        return x

    def _get_pert_model_forward_fn(self) -> Callable:
        def forward_fn(inp: torch.Tensor) -> torch.Tensor:
            batch_size = inp.size(0)
            self.model.eval()
            logit = self.get_forward_fn()(inp.view(batch_size, -1))
            return logit

        return forward_fn

    def _get_pert_model_loss_fn(self) -> Callable:
        def head_loss_fn(x: torch.Tensor):
            self.head_model.eval()
            loss_fn = self.get_per_sample_eval_fn()
            return loss_fn(x)

        return head_loss_fn

    def train(self):
        self.set_model_state(train=False)

        # eval before perturbation
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")

        # start perturbating
        x = self._get_pert_model_dataset(self.dl)
        x_eval = self._get_pert_model_dataset(self.eval_dl)

        # update perturbation optimizaer, and clean up model's gradient
        self.set_model_state(train=False)
        self.pert_model.update(
            x=x,
            x_eval=x_eval,
            forward_fn=self._get_pert_model_forward_fn(),
            device=self.device,
            head_loss_fn=self._get_pert_model_loss_fn(),
            mask_value=self.pert_model.get_pert_val()
            .detach()
            .clone()
            .reshape(28, 28)[None, None, :, :],
        )

        self.pert_step += 1
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")
        log_results = self.eval(self.test_dl)
        self.log(log_results, section="test")

        self.save("final")
        print("perturbation complete")


class MNISTLIMEPerturbationTrainer(MNISTPerturbationTrainer):
    def _get_pert_model_dataset(self, dl: DataLoader) -> torch.Tensor:
        xs = []
        for data, *_ in dl:
            data = data.reshape(-1, 28, 28).to(self.device)
            xs.append(data.unsqueeze(1))
        x = torch.cat(xs, dim=0)
        return x

    def _get_pert_model_forward_fn(self) -> Callable:
        def forward_fn(inp: torch.Tensor) -> torch.Tensor:
            batch_size = inp.size(0)
            self.model.eval()
            self.head_model.eval()
            loss_fn = self.get_per_sample_eval_fn()
            logit = self.get_forward_fn()(inp.view(batch_size, -1))
            return loss_fn(logit)

        return forward_fn

    def train(self):
        self.set_model_state(train=False)

        # eval before perturbation
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")

        # start perturbating
        x = self._get_pert_model_dataset(self.dl)

        # update perturbation optimizaer, and clean up model's gradient
        self.set_model_state(train=False)
        self.pert_model.update(
            x=x,
            forward_fn=self._get_pert_model_forward_fn(),
            mask_value=self.pert_model.get_pert_val()
            .detach()
            .clone()
            .reshape(28, 28)[None, None, :, :],
            label=0,
        )

        self.pert_step += 1
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")
        log_results = self.eval(self.test_dl)
        self.log(log_results, section="test")

        self.save("final")
        print("perturbation complete")
