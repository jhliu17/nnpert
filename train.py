import torch

from src.model.cross_mapping import AtacRnaCrossMappingConfig, AtacRnaCrossMappingModel
from src.trainer_brain import Trainer, TrainerConfig
from src.wandb import WandbConfig, init_wandb
from src.slurm import SlurmConfig, slurm_launcher
from dataclasses import asdict, dataclass, field
from src.utils import seed_everything


@dataclass
class ExperimentConfig:
    # model configurations
    model: AtacRnaCrossMappingConfig = field(default_factory=AtacRnaCrossMappingConfig)

    # trainer arguments
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    # wandb
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # slurm
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    # trainer should be resumed
    resume_trainer: bool = False

    # trainer checkpoint path
    trainer_ckpt_path: str = ""

    # resume from pretrained model
    train_affine: bool = False

    # pretrained model path
    pretrained_trainer_ckpt_path: str = ""


@slurm_launcher(ExperimentConfig)
def main(args: ExperimentConfig):
    seed_everything(args.trainer.seed)
    init_wandb(args.wandb, asdict(args))

    model = AtacRnaCrossMappingModel(args.model)
    trainer = Trainer(model, args=args.trainer, has_wandb_writer=True)
    if args.resume_trainer:
        state_pt = torch.load(args.trainer_ckpt_path, map_location=trainer.device)
        trainer.load_state(state_pt)

    if args.train_affine:
        state_pt = torch.load(
            args.pretrained_trainer_ckpt_path, map_location=trainer.device
        )
        trainer.load_state(state_pt)
        trainer.train_affine()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
