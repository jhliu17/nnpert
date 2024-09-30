import torch
import torch.nn as nn

from dataclasses import asdict, dataclass, field
from src.model.classifier import FFNClassifier
from src.train.mnist import (
    MNISTClassifierTrainerConfig,
    MNISTClassifierTrainer,
)
from src.trainer import (
    MNISTPerturbationTrainer,
    MNISTPerturbationTrainerConfig,
    MNISTSAGEPerturbationTrainer,
    MNISTCXPlainPerturbationTrainer,
    MNISTLIMEPerturbationTrainer,
)
from src.wandb import WandbConfig, init_wandb
from src.slurm import SlurmConfig, slurm_launcher
from src.pert import PertModuleConfig, PertModule
from src.dispatch import dispatch_pert_module
from src.head.cls import NeuralClassifierHead
from src.utils import seed_everything


@dataclass
class ExperimentArgs:
    # trainer should be resumed
    resume_trainer: bool = False

    # trainer method name
    trainer_method: str = "perturb"

    # pert model configurations
    pert: PertModuleConfig = field(default_factory=PertModuleConfig)

    # head model configurations
    head_trainer: MNISTClassifierTrainerConfig = field(
        default_factory=MNISTClassifierTrainerConfig
    )

    # trainer arguments
    trainer: MNISTPerturbationTrainerConfig = field(
        default_factory=MNISTPerturbationTrainerConfig
    )

    # wandb
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # slurm
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    # random seed
    seed: int = 2024


@slurm_launcher(ExperimentArgs)
def main(args: ExperimentArgs):
    seed_everything(args.seed)
    init_wandb(args.wandb, asdict(args))

    # model input
    input_dim = 28 * 28

    # train head model
    head_layer = FFNClassifier(
        input_dim,
        len(args.head_trainer.num_type_list),
        hidden_dim=args.head_trainer.hidden_dim,
    )
    head_trainer = MNISTClassifierTrainer(
        head_layer, args=args.head_trainer, has_wandb_writer=True
    )
    head_trainer.train()

    head_model = NeuralClassifierHead(
        head_layer,
        tgt_ind=args.head_trainer.num_type_list.index(args.head_trainer.tgt_num_type),
    )

    # init pert model
    pert_val = torch.zeros(input_dim)
    pert_model: PertModule = dispatch_pert_module(
        pert_type=args.pert.model_type,
        pert_num=args.pert.perturbation_num,
        pert_space=input_dim,
        pert_val=pert_val,
        pert_step=args.trainer.pert_num_steps,
        pert_config=args.pert,
    )

    # init trainer methods
    trainer_method = {
        "perturb": MNISTPerturbationTrainer,
        "sage": MNISTSAGEPerturbationTrainer,
        "cxplain": MNISTCXPlainPerturbationTrainer,
        "lime": MNISTLIMEPerturbationTrainer,
    }

    # init surrogate and start perturbating
    model = nn.Identity()
    trainer_cls = trainer_method[args.trainer_method]
    trainer = trainer_cls(
        pert_model,
        model,
        head_model,
        head_trainer.num_type_dict,
        args=args.trainer,
        has_wandb_writer=True,
    )

    trainer.train()


if __name__ == "__main__":
    main()
