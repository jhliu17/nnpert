import torch
import torch.nn as nn
import wandb
import scanpy as sc
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List, Literal, Tuple, Callable
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from dataclasses import dataclass, asdict, field
from itertools import chain
from sklearn.preprocessing import LabelEncoder

from .utils import (
    divisible_by,
    cycle_dataloader,
)
from .head.base import HeadModule
from .dataset import SequenceDataSet
from .pert import PertModule
from .train.base import BaseTrainer
from .model.cross_mapping import AtacRnaCrossMappingModel
from .model.op import masked_bce, masked_mse
from .aurora.umap.visualization import get_joint_umap_figure, get_single_umap_figure
from .aurora.plot import plot_gene_expr
from .aurora.eval import eval_search_performance


@dataclass
class TrainerConfig:
    # dataset name
    dataset_name: str

    # output path
    output_folder: str

    atac_seq_path: str

    atac_seq_label_path: str

    rna_seq_path: str

    rna_seq_label_path: str

    # optimization
    train_batch_size: int = 16

    train_atac_ae_lr: float = 1e-4

    train_rna_ae_lr: float = 1e-4

    train_ae_lr: float = 1e-4

    train_aff_lr: float = 1e-4

    train_gen_lr: float = 1e-4

    train_dis_lr: float = 1e-4

    train_atac_ae_num_steps: int = 3000

    train_rna_ae_num_steps: int = 3000

    train_affine_num_steps: int = 3000

    save_and_sample_every: int = 1000

    max_grad_norm: float = 1

    update_dis_freq: int = 3

    adam_betas: Tuple[float, float] = (0.9, 0.99)

    lamda1: float = 1.0

    lamda2: float = 5.0

    lamda3: float = 1.0

    lamda4: float = 1.0

    focal_gamma: float = 1.0

    # device
    num_workers: int = 0

    use_cuda: bool = True

    # random seed
    seed: int = 2023


class Trainer(BaseTrainer):
    def __init__(
        self,
        model: AtacRnaCrossMappingModel,
        *,
        args: TrainerConfig,
        has_wandb_writer: bool = False,
    ) -> None:
        super().__init__(args.output_folder, has_wandb_writer=has_wandb_writer)

        self.args = args
        self.batch_size = args.train_batch_size
        self.seed = args.seed
        self.dataset_name = args.dataset_name

        # set dataloaders
        self.rna_cls_label, self.rna_cls_label_encoder = self.read_seq_label(
            args.rna_seq_label_path
        )
        self.atac_cls_label, _ = self.read_seq_label(args.atac_seq_label_path)

        atac_seq_dataset = self.build_seq_dataset(
            args.atac_seq_path,
            self.atac_cls_label,
            self.rna_cls_label_encoder,
            do_normalization=False,
        )
        rna_seq_dataset = self.build_seq_dataset(
            args.rna_seq_path,
            self.rna_cls_label,
            self.rna_cls_label_encoder,
            do_normalization=True,
        )
        self.atac_dl = cycle_dataloader(
            DataLoader(
                atac_seq_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
        )
        self.rna_dl = cycle_dataloader(
            DataLoader(
                rna_seq_dataset,
                shuffle=True,
                batch_size=self.batch_size,
                num_workers=args.num_workers,
            )
        )
        self.test_atac_dl = DataLoader(
            atac_seq_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        self.test_rna_dl = DataLoader(
            rna_seq_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        # step counter state
        self.step = 0
        self.train_atac_ae_num_steps = args.train_atac_ae_num_steps
        self.train_rna_ae_num_steps = (
            args.train_rna_ae_num_steps + self.train_atac_ae_num_steps
        )
        self.train_affine_num_steps = (
            args.train_affine_num_steps + self.train_rna_ae_num_steps
        )
        self.save_and_sample_every = args.save_and_sample_every

        # model and optimizer
        self.model = model.to(self.device)
        self.max_grad_norm = args.max_grad_norm
        self.opt_atac = Adam(
            self.model.atac_model.parameters(),
            lr=args.train_atac_ae_lr,
            betas=args.adam_betas,
        )
        self.opt_rna = Adam(
            self.model.rna_model.parameters(),
            lr=args.train_rna_ae_lr,
            betas=args.adam_betas,
        )
        self.opt_ae = Adam(
            chain(*[m.parameters() for m in self.model.autoencoders]),
            lr=args.train_ae_lr,
            betas=args.adam_betas,
        )
        self.opt_aff = Adam(
            chain(*[m.parameters() for m in self.model.affine_transforms]),
            lr=args.train_aff_lr,
            betas=args.adam_betas,
        )
        self.opt_gen = Adam(
            chain(*[m.parameters() for m in self.model.generators]),
            lr=args.train_gen_lr,
            betas=args.adam_betas,
        )
        self.opt_dis = Adam(
            chain(*[m.parameters() for m in self.model.discriminators]),
            lr=args.train_dis_lr,
            betas=args.adam_betas,
        )
        self.update_dis_freq = args.update_dis_freq
        self.lamda1 = args.lamda1
        self.lamda2 = args.lamda2
        self.lamda3 = args.lamda3
        self.lamda4 = args.lamda4
        self.focal_gamma = args.focal_gamma

    def read_seq_label(self, seq_label_path):
        label_frame = pd.read_csv(seq_label_path, delimiter="\t", header=None)
        labels = label_frame.iloc[:, 1].tolist()

        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        return labels, label_encoder

    def build_seq_dataset(
        self,
        seq_path: str,
        seq_cls_labels: list[str],
        seq_cls_encoder: LabelEncoder,
        do_normalization: bool = False,
    ):
        seq_dataset = SequenceDataSet(
            seq_path, seq_cls_labels, seq_cls_encoder, do_normalization=do_normalization
        )
        return seq_dataset

    def get_state(self):
        state = {
            "step": self.step,
            "model": self.model.state_dict(),
            "opt_atac": self.opt_atac.state_dict(),
            "opt_rna": self.opt_rna.state_dict(),
            "opt_ae": self.opt_ae.state_dict(),
            "opt_aff": self.opt_aff.state_dict(),
            "opt_gen": self.opt_gen.state_dict(),
            "opt_dis": self.opt_dis.state_dict(),
            "args": asdict(self.args),
        }
        return state

    def load_state(self, state):
        self.model.load_state_dict(state["model"])
        self.step = state["step"]
        self.opt_atac.load_state_dict(state["opt_atac"])
        self.opt_rna.load_state_dict(state["opt_rna"])
        self.opt_ae.load_state_dict(state["opt_ae"])
        self.opt_aff.load_state_dict(state["opt_aff"])
        self.opt_gen.load_state_dict(state["opt_gen"])
        self.opt_dis.load_state_dict(state["opt_dis"])

    @property
    def device(self):
        return torch.device("cuda") if self.args.use_cuda else torch.device("cpu")

    @property
    def global_step(self) -> int:
        return self.step

    def set_model_state(self, train: bool = True, part: str = "all"):
        """In this trainer, we only train on the model.

        :param train: whether to train mode, defaults to True
        """
        if part == "all":
            self.model.train(train)
        else:
            for m in getattr(self.model, part):
                m.train(train)

    @torch.inference_mode()
    def generate_latent(self, atac_dl: DataLoader, rna_dl: DataLoader, part: str):
        self.set_model_state(train=False)
        if part == "atac":
            latent = []
            for data, *_ in atac_dl:
                data = data.to(self.device)
                latent.append(self.model.encode(data, "atac"))
            latent = torch.cat(latent, dim=0)
            return latent
        elif part == "rna":
            latent = []
            for data, *_ in rna_dl:
                data = data.to(self.device)
                latent.append(self.model.encode(data, "rna"))
            latent = torch.cat(latent, dim=0)
            return latent
        elif part == "affine":
            atac_latent = []
            rna_latent = []

            atac2rna_latent = []
            rna2atac_latent = []

            for data, *_ in atac_dl:
                data = data.to(self.device)
                latent = self.model.encode(data, "atac")
                map_latent = self.model.transform(latent, "atac2rna")
                atac_latent.append(latent)
                atac2rna_latent.append(map_latent)

            for data, *_ in rna_dl:
                data = data.to(self.device)
                latent = self.model.encode(data, "rna")
                map_latent = self.model.transform(latent, "rna2atac")
                rna_latent.append(latent)
                rna2atac_latent.append(map_latent)

            atac_latent = torch.cat(atac_latent, dim=0)
            rna_latent = torch.cat(rna_latent, dim=0)
            atac2rna_latent = torch.cat(atac2rna_latent, dim=0)
            rna2atac_latent = torch.cat(rna2atac_latent, dim=0)
            return atac_latent, rna_latent, atac2rna_latent, rna2atac_latent
        else:
            raise ValueError(f"Unknown part: {part}")

    @torch.inference_mode()
    def eval(self, atac_dl: DataLoader, rna_dl: DataLoader, part: str):
        self.set_model_state(train=False)
        umap_config = {
            "random_state": self.seed,
            "n_neighbors": 30,
            "min_dist": 0.3,
            "n_components": 2,
            "metric": "cosine",
        }

        if part in {"atac", "rna"}:
            latent = self.generate_latent(atac_dl, rna_dl, part)
            latent = latent.numpy(force=True)
            figure_dict, score_dict = get_single_umap_figure(
                latent, latent, self.rna_cls_label, umap_config, source=part
            )
        elif part == "affine":
            (
                atac_latent,
                rna_latent,
                atac2rna_latent,
                rna2atac_latent,
            ) = self.generate_latent(atac_dl, rna_dl, part)
            figure_dict, score_dict = get_joint_umap_figure(
                atac2rna_latent.numpy(force=True),
                atac_latent.numpy(force=True),
                rna_latent.numpy(force=True),
                rna2atac_latent.numpy(force=True),
                self.rna_cls_label,
                umap_config,
            )
        else:
            raise ValueError(f"Unknown part: {part}")

        # post processing
        results = {}
        for k, v in figure_dict.items():
            results[k] = wandb.Image(v)
        results.update(score_dict)
        return results

    @torch.inference_mode()
    def eval_during_training(self, part: str):
        results = self.eval(self.test_atac_dl, self.test_rna_dl, part)
        self.set_model_state(train=True)
        return results

    def train_atac_ae(self):
        self.set_model_state(train=True)

        device = self.device

        with tqdm(
            initial=self.step,
            total=self.train_atac_ae_num_steps,
        ) as pbar:
            while self.step < self.train_atac_ae_num_steps:
                data, *_ = next(self.atac_dl)
                data = data.to(device)

                recon = self.model(data, mode="atac")
                loss = masked_bce(recon, data, gamma=self.focal_gamma)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.atac_model.parameters(), max_norm=self.max_grad_norm
                )

                self.opt_atac.step()
                self.opt_atac.zero_grad()
                pbar.set_description(f"loss: {loss.item():.4f}")
                self.log({"loss": loss.item()}, section="train_atac_ae")

                self.step += 1
                if self.step != 0 and divisible_by(
                    self.step, self.save_and_sample_every
                ):
                    log_results = self.eval_during_training("atac")
                    self.log(log_results, section="eval_atac_ae")
                    milestone = self.step // self.save_and_sample_every
                    self.save(milestone)

                pbar.update(1)

        self.save("final")
        log_results = self.eval(self.test_atac_dl, None, "atac")
        self.log(log_results, section="test_atac_ae")
        print("atac autoencoder training complete")

    def train_rna_ae(self):
        self.set_model_state(train=True)

        device = self.device
        loss_fn = masked_mse

        with tqdm(
            initial=self.step,
            total=self.train_rna_ae_num_steps,
        ) as pbar:
            while self.step < self.train_rna_ae_num_steps:
                data, *_ = next(self.rna_dl)
                data = data.to(device)

                recon = self.model(data, mode="rna")
                loss = loss_fn(recon, data)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.rna_model.parameters(), max_norm=self.max_grad_norm
                )

                self.opt_rna.step()
                self.opt_rna.zero_grad()
                pbar.set_description(f"loss: {loss.item():.4f}")
                self.log({"loss": loss.item()}, section="train_rna_ae")

                self.step += 1
                if self.step != 0 and divisible_by(
                    self.step, self.save_and_sample_every
                ):
                    log_results = self.eval_during_training("rna")
                    self.log(log_results, section="eval_rna_ae")
                    milestone = self.step // self.save_and_sample_every
                    self.save(milestone)

                pbar.update(1)

        self.save("final")
        log_results = self.eval(None, self.test_rna_dl, "rna")
        self.log(log_results, section="test_rna_ae")
        print("rna autoencoder training complete")

    def compute_affine_loss(self, atac_seq, atac_seq_cls, rna_seq, rna_seq_cls, step):
        # self.set_model_state(train=False, part="autoencoders")

        # encode
        atac_latent = self.model.encode(atac_seq, "atac")
        rna_latent = self.model.encode(rna_seq, "rna")

        # cross mapping
        atac2rna_latent = self.model.transform(atac_latent, "atac2rna")
        rna2atac_latent = self.model.transform(rna_latent, "rna2atac")
        recon_atac_latent = self.model.transform(atac2rna_latent, "rna2atac")
        recon_rna_latent = self.model.transform(rna2atac_latent, "atac2rna")

        # feature reconstruction
        recon_atac_seq = self.model.decode(recon_atac_latent, "atac")
        recon_rna_seq = self.model.decode(recon_rna_latent, "rna")

        # latent cycle loss
        rna_latent_cycle_loss = F.l1_loss(rna_latent, recon_rna_latent)
        atac_latent_cycle_loss = F.l1_loss(atac_latent, recon_atac_latent)

        # reconstruction cycle loss
        recon_atac_seq_cycle_loss = masked_bce(
            recon_atac_seq, atac_seq, gamma=self.focal_gamma
        )
        recon_rna_seq_cycle_loss = masked_mse(recon_rna_seq, rna_seq)

        # total recon loss (cycle)
        recon_loss = self.lamda1 * (
            rna_latent_cycle_loss / self.lamda2 + atac_latent_cycle_loss
        ) + (recon_rna_seq_cycle_loss / self.lamda2 + recon_atac_seq_cycle_loss)

        # discriminator loss
        atac_batch_size = atac_seq.size(0)
        rna_batch_size = rna_seq.size(0)
        atac_real = torch.ones(atac_batch_size, 1).to(self.device)
        atac_fake = torch.zeros(atac_batch_size, 1).to(self.device)
        rna_real = torch.ones(rna_batch_size, 1).to(self.device)
        rna_fake = torch.zeros(rna_batch_size, 1).to(self.device)

        adv_loss_fn = F.binary_cross_entropy_with_logits
        gen2dis_loss = (
            adv_loss_fn(
                self.model(atac2rna_latent, atac_seq_cls, mode="rna_dis"), atac_real
            )
            + adv_loss_fn(
                self.model(recon_atac_latent, atac_seq_cls, mode="atac_dis"), atac_real
            )
            + adv_loss_fn(
                self.model(rna2atac_latent, rna_seq_cls, mode="atac_dis"), rna_real
            )
            + adv_loss_fn(
                self.model(recon_rna_latent, rna_seq_cls, mode="rna_dis"), rna_real
            )
        )

        gen_loss = recon_loss + self.lamda3 * gen2dis_loss
        self.opt_ae.zero_grad()
        self.opt_aff.zero_grad()
        gen_loss.backward()
        self.opt_aff.step()

        log_dict = {}
        log_dict["gen/cross_recon"] = recon_loss.item()
        log_dict["gen/gen2dis_loss"] = gen2dis_loss.item()
        log_dict["gen/loss"] = gen_loss.item()

        sub_log_dict = {}
        sub_log_dict["gen/rna_latent_cycle_loss"] = rna_latent_cycle_loss.item()
        sub_log_dict["gen/atac_latent_cycle_loss"] = atac_latent_cycle_loss.item()
        sub_log_dict["gen/recon_rna_seq_cycle_loss"] = recon_rna_seq_cycle_loss.item()
        sub_log_dict["gen/recon_atac_seq_cycle_loss"] = recon_atac_seq_cycle_loss.item()

        if step % self.update_dis_freq == 0:
            # discriminator loss
            rna_dis_loss = adv_loss_fn(
                self.model(rna_latent.detach(), rna_seq_cls, mode="rna_dis"), rna_real
            ) + adv_loss_fn(
                self.model(atac2rna_latent.detach(), atac_seq_cls, mode="rna_dis"),
                atac_fake,
            )
            atac_dis_loss = adv_loss_fn(
                self.model(atac_latent.detach(), atac_seq_cls, mode="atac_dis"),
                atac_real,
            ) + adv_loss_fn(
                self.model(rna2atac_latent.detach(), rna_seq_cls, mode="atac_dis"),
                rna_fake,
            )

            dis_loss = (rna_dis_loss + atac_dis_loss) * self.lamda4
            self.opt_dis.zero_grad()
            dis_loss.backward()
            self.opt_dis.step()
            log_dict["dis/loss"] = dis_loss.item()

        return log_dict, sub_log_dict

    def train_affine(self):
        self.set_model_state(train=True)

        device = self.device

        with tqdm(
            initial=self.step,
            total=self.train_affine_num_steps,
        ) as pbar:
            while self.step < self.train_affine_num_steps:
                atac_seq, atac_seq_cls, _ = next(self.atac_dl)
                rna_seq, rna_seq_cls, _ = next(self.rna_dl)

                atac_seq, atac_seq_cls = atac_seq.to(device), atac_seq_cls.to(device)
                rna_seq, rna_seq_cls = rna_seq.to(device), rna_seq_cls.to(device)

                loss_dict, loss_details_dict = self.compute_affine_loss(
                    atac_seq, atac_seq_cls, rna_seq, rna_seq_cls, self.step
                )

                self.log(loss_dict, section="train_affine")
                self.log(loss_details_dict, section="train_affine_details")

                self.step += 1
                if self.step != 0 and divisible_by(
                    self.step, self.save_and_sample_every
                ):
                    log_results = self.eval_during_training("affine")
                    self.log(log_results, section="eval_affine")
                    milestone = self.step // self.save_and_sample_every
                    self.save(milestone)

                pbar.update(1)

        self.save("final")
        log_results = self.eval(self.test_atac_dl, self.test_rna_dl, "affine")
        self.log(log_results, section="test_affine")
        print("training complete")

    def train(self):
        self.set_model_state(train=True)

        self.train_atac_ae()
        self.train_rna_ae()
        self.train_affine()


@dataclass
class PerturbationTask:
    target_cell_type: str

    target_gene: str

    target_gene_ind: int


@dataclass
class PerturbationTrainerConfig(TrainerConfig):
    # rna perturbation direction
    rna_expression_direction: Literal["down", "up"] = "down"

    # atac perturbation direction
    atac_expression_direction: Literal["down", "up"] = "down"

    gene_region_file: str = ""

    atac_description_file: str = ""

    rna_description_file: str = ""

    # perturbation total step
    pert_num_steps: int = 100000

    # perturbation state step
    pert_state_step: int = -1

    eval_trial_size: int = 1

    # train test split ratio
    train_test_split_ratio: list[float] = field(default_factory=lambda: [0.7, 0.1, 0.2])

    # compliment target cell type
    complement_target_cell_type: bool = False


class PerturbationTrainer(Trainer):
    def __init__(
        self,
        pert_model: PertModule,
        model: nn.Module,
        head_model: HeadModule,
        task: PerturbationTask,
        *,
        args: PerturbationTrainerConfig,
        has_wandb_writer: bool = False,
    ) -> None:
        super().__init__(model, args=args, has_wandb_writer=has_wandb_writer)
        self.pert_args = args

        self.pert_model = pert_model.to(self.device)
        self.head_model = head_model.to(self.device)

        self.target_cell_type = task.target_cell_type
        self.target_gene = task.target_gene
        self.target_gene_ind = task.target_gene_ind
        self.gene_region_file = args.gene_region_file
        self.atac_description_file = args.atac_description_file
        self.train_test_split_ratio = args.train_test_split_ratio
        self.eval_trial_size = args.eval_trial_size

        # dummy optimizer for clean gradient on head model
        # self.head_model_opt = SGD(self.head_model.parameters(), lr=1e-3)

        # build target dataset
        train_ds, eval_ds, test_ds = self.build_pert_dataset(
            self.target_cell_type,
            args.complement_target_cell_type,
            args.atac_seq_path,
            self.rna_cls_label,
            self.rna_cls_label_encoder,
            do_normalization=False,
        )
        self.pert_atac_dl = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        self.eval_pert_atac_dl = DataLoader(
            eval_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        self.test_pert_atac_dl = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        # step counter state
        self.pert_step = 0
        self.pert_num_steps = self.pert_args.pert_num_steps

    def build_pert_dataset(
        self,
        target_cell_type: str,
        complement_target_cell_type: bool,
        atac_seq_path: str,
        atac_cls_label: list[str],
        rna_cls_label_encoder: LabelEncoder,
        do_normalization: bool = False,
    ):
        dataset = SequenceDataSet(
            atac_seq_path,
            atac_cls_label,
            rna_cls_label_encoder,
            do_normalization=do_normalization,
            target_cell_type=target_cell_type,
            complement_target_cell_type=complement_target_cell_type,
        )

        generator = torch.Generator().manual_seed(self.seed)
        train_ds, eval_ds, test_ds = random_split(
            dataset,
            self.train_test_split_ratio,
            generator=generator,
        )
        return train_ds, eval_ds, test_ds

    def get_state(self):
        parent_state = super().get_state()
        state = {
            "pert_step": self.pert_step,
            "pert_model": self.pert_model.state_dict(),
            "pert_args": asdict(self.pert_args),
            "head_model": self.head_model.state_dict(),
            "parent_state": parent_state,
        }
        return state

    def load_state(self, state):
        super().load_state(state["parent_state"])
        self.pert_step = state["pert_step"]
        self.pert_model.load_state_dict(state["pert_model"])
        self.head_model.load_state_dict(state["head_model"])

    @property
    def global_step(self) -> int:
        return self.pert_step

    def get_forward_fn(self):
        def forward_fn(inp: torch.Tensor) -> torch.Tensor:
            pred = self.model.encode_given_embed(inp, mode="atac")
            pred = self.model.decode(self.model.transform(pred, "atac2rna"), mode="rna")
            final_state = torch.clamp(pred, 0)
            return final_state

        return forward_fn

    def get_per_sample_eval_fn(self):
        @torch.inference_mode()
        def eval_fn(final_state: torch.Tensor) -> torch.Tensor:
            loss = self.head_model.get_loss(final_state, reduction="none")
            return loss

        return eval_fn

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
    def generate_cross_mapping(
        self, atac_dl: DataLoader, pert_model: PertModule = None
    ):
        self.set_model_state(train=False)
        rna_expr = []
        for data, *_ in atac_dl:
            data = data.to(self.device)
            if pert_model is None:
                rna_expr.append(torch.clamp(self.model(data, mode="atac2rna"), 0))
            else:
                pert_data = pert_model.perturb(data)
                rna_expr.append(self.get_forward_fn()(pert_data))
        rna_expr = torch.cat(rna_expr, dim=0)
        return rna_expr

    @torch.inference_mode()
    def eval_changed_value(
        self, rna_expr: torch.Tensor, pert_rna_expr: torch.Tensor, target_gene_ind: int
    ):
        origin = rna_expr[:, target_gene_ind].exp()
        after = pert_rna_expr[:, target_gene_ind].exp()

        abs_diff = after - origin
        rel_diff = abs_diff / origin

        abs_avg = abs_diff.mean().item()
        rel_avg = rel_diff.mean().item()

        return {
            "abs_avg": abs_avg,
            "rel_avg": rel_avg,
        }

    @torch.inference_mode()
    def eval_search_performance(self):
        self.set_model_state(train=False)
        pert_ind = self.pert_model.get_pert_ind()

        # eval search performance
        search_metrics = eval_search_performance(
            self.gene_region_file,
            self.atac_description_file,
            self.target_gene,
            pert_ind.numpy(force=True),
        )
        return search_metrics

    @torch.inference_mode()
    def eval(self, dl: DataLoader):
        self.set_model_state(train=False)
        rna_expr = self.generate_cross_mapping(dl)
        pert_rna_expr = self.generate_cross_mapping(dl, self.pert_model)

        results = self.eval_changed_value(rna_expr, pert_rna_expr, self.target_gene_ind)

        # eval search performance
        search_metrics = self.eval_search_performance()
        results.update(search_metrics)

        fig, _ = plot_gene_expr(
            f"{self.target_cell_type}: {self.target_gene}",
            rna_expr.numpy(force=True)[:, self.target_gene_ind],
            pert_rna_expr.numpy(force=True)[:, self.target_gene_ind],
            bins="auto",
        )
        results.update({"gene_expr_dist": wandb.Image(fig)})
        return results

    @torch.inference_mode()
    def eval_during_training(self, dl: DataLoader = None):
        results = self.eval(self.eval_pert_atac_dl if dl is None else dl)
        self.set_model_state(train=True)
        return results

    def train(self):
        self.set_model_state(train=True)

        # eval before perturbation
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")

        # start perturbating
        dl = cycle_dataloader(self.pert_atac_dl)
        with tqdm(
            initial=self.pert_step,
            total=self.pert_num_steps,
        ) as pbar:
            while self.pert_step < self.pert_num_steps:
                data = next(dl)
                inp = data[0].to(self.device)

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
                    eval_trial_size=self.eval_trial_size,
                )
                self.model.zero_grad()
                self.head_model.zero_grad()

                # eval search performance
                log_results = self.eval_search_performance()
                self.log(log_results, section="eval")
                self.set_model_state(train=True)

                self.pert_step += 1
                if self.pert_step != 0 and divisible_by(
                    self.pert_step, self.save_and_sample_every
                ):
                    log_results = self.eval_during_training()
                    self.log(log_results, section="eval")
                    log_results = self.eval_during_training(self.test_pert_atac_dl)
                    self.log(log_results, section="test")
                    milestone = self.pert_step // self.save_and_sample_every
                    self.save(milestone)

                pbar.update(1)

        self.save("final")
        log_results = self.eval(self.test_pert_atac_dl)
        self.log(log_results, section="test")
        print("perturbation complete")


class AuroraSAGEPerturbationTrainer(PerturbationTrainer):
    def train(self):
        self.set_model_state(train=True)

        # eval before perturbation
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")

        def forward_fn(inp: torch.Tensor) -> torch.Tensor:
            self.set_model_state(train=False)
            return torch.clamp(self.model(inp, mode="atac2rna"), 0)

        # start perturbating
        xs = []
        ys = []
        for data, *_ in self.pert_atac_dl:
            data = data.to(self.device)
            xs.append(data)
            ys.append(forward_fn(data))
        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)

        # update perturbation optimizaer, and clean up model's gradient
        self.set_model_state(train=False)
        self.pert_model.update(
            x=x,
            y=y,
            forward_fn=forward_fn,
            device=self.device,
            head_loss_fn=self.head_model.get_loss,
        )

        self.pert_step += 1
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")
        log_results = self.eval(self.test_pert_atac_dl)
        self.log(log_results, section="test")

        self.save("final")
        print("perturbation complete")


class AuroraCXPlainPerturbationTrainer(PerturbationTrainer):
    def _get_pert_model_dataset(self, dl: DataLoader) -> torch.Tensor:
        xs = []
        for data, *_ in dl:
            data = data.to(self.device)
            xs.append(data)
        x = torch.cat(xs, dim=0)
        return x

    def _get_pert_model_forward_fn(self) -> Callable:
        def forward_fn(inp: torch.Tensor) -> torch.Tensor:
            self.set_model_state(train=False)
            return torch.clamp(self.model(inp, mode="atac2rna"), 0)

        return forward_fn

    def _get_pert_model_loss_fn(self) -> Callable:
        def head_loss_fn(x: torch.Tensor):
            self.head_model.eval()
            loss_fn = self.get_per_sample_eval_fn()
            return loss_fn(x)

        return head_loss_fn

    def train(self):
        self.set_model_state(train=True)

        # eval before perturbation
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")

        # start perturbating
        x = self._get_pert_model_dataset(self.pert_atac_dl)
        x_eval = self._get_pert_model_dataset(self.eval_pert_atac_dl)

        # update perturbation optimizaer, and clean up model's gradient
        self.set_model_state(train=False)
        self.pert_model.update(
            x=x,
            x_eval=x_eval,
            forward_fn=self._get_pert_model_forward_fn(),
            device=self.device,
            head_loss_fn=self._get_pert_model_loss_fn(),
            mask_value=self.pert_model.get_pert_val().detach().clone()[None, :],
        )

        self.pert_step += 1
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")
        log_results = self.eval(self.test_pert_atac_dl)
        self.log(log_results, section="test")

        self.save("final")
        print("perturbation complete")


class AuroraLIMEPerturbationTrainer(PerturbationTrainer):
    def _get_pert_model_dataset(self, dl: DataLoader) -> torch.Tensor:
        xs = []
        for data, *_ in dl:
            data = data.to(self.device)
            xs.append(data)
        x = torch.cat(xs, dim=0)
        return x

    def _get_pert_model_forward_fn(self) -> Callable:
        def forward_fn(inp: torch.Tensor) -> torch.Tensor:
            self.set_model_state(train=False)
            loss_fn = self.get_per_sample_eval_fn()
            gene_expr = torch.clamp(self.model(inp, mode="atac2rna"), 0)
            return loss_fn(gene_expr)

        return forward_fn

    def train(self):
        self.set_model_state(train=True)

        # eval before perturbation
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")

        # start perturbating
        x = self._get_pert_model_dataset(self.pert_atac_dl)

        # update perturbation optimizaer, and clean up model's gradient
        self.set_model_state(train=False)
        self.pert_model.update(
            x=x,
            forward_fn=self._get_pert_model_forward_fn(),
            mask_value=self.pert_model.get_pert_val().detach().clone()[None, :],
            label=0,
        )

        self.pert_step += 1
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")
        log_results = self.eval(self.test_pert_atac_dl)
        self.log(log_results, section="test")

        self.save("final")
        print("perturbation complete")
