import torch
import copy
import wandb

from dataclasses import asdict, dataclass, field
from src.model.cross_mapping import AtacRnaCrossMappingConfig, AtacRnaCrossMappingModel
from src.trainer_brain import (
    PerturbationTrainer,
    PerturbationTrainerConfig,
    PerturbationTask,
    AuroraSAGEPerturbationTrainer,
    AuroraCXPlainPerturbationTrainer,
    AuroraLIMEPerturbationTrainer,
)
from src.wandb import WandbConfig, init_wandb
from src.slurm import SlurmConfig, slurm_launcher
from src.pert import PertModuleConfig, PertModule
from src.dispatch import dispatch_pert_module
from src.head.val import ValueIndexerHead
from src.utils import seed_everything
from src.aurora.utils import load_gene_list


@dataclass
class ExperimentArgs:
    pert_target_cell_type: list[str]

    pert_target_gene: list[str]

    # pretrained trainer checkpoint path
    pretrained_trainer_ckpt_path: str

    pert_target_gene_file: str = ""

    # trainer method name
    trainer_method: str = "perturb"

    # trainer checkpoint path
    trainer_ckpt_path: str = ""

    # trainer should be resumed
    resume_trainer: bool = False

    # model configurations
    model: AtacRnaCrossMappingConfig = field(default_factory=AtacRnaCrossMappingConfig)

    # pert model configurations
    pert: PertModuleConfig = field(default_factory=PertModuleConfig)

    # trainer arguments
    trainer: PerturbationTrainerConfig = field(
        default_factory=PerturbationTrainerConfig
    )

    # wandb
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # slurm
    slurm: SlurmConfig = field(default_factory=SlurmConfig)


@slurm_launcher(ExperimentArgs)
def main(args: ExperimentArgs):
    seed_everything(args.trainer.seed)

    # init model
    model = AtacRnaCrossMappingModel(args.model)
    state_pt = torch.load(args.pretrained_trainer_ckpt_path)
    model.load_state_dict(state_pt["model"])

    # init pert model
    pert_val = (
        torch.zeros(args.model.atac_model.input_dim)
        if args.trainer.atac_expression_direction == "down"
        else torch.ones(args.model.atac_model.input_dim)
    )

    # init gene list
    gene_list = load_gene_list(args.trainer.rna_description_file)

    # init trainer methods
    trainer_method = {
        "perturb": PerturbationTrainer,
        "sage": AuroraSAGEPerturbationTrainer,
        "cxplain": AuroraCXPlainPerturbationTrainer,
        "lime": AuroraLIMEPerturbationTrainer,
    }

    # searching loop
    if len(args.pert_target_cell_type) != len(args.pert_target_gene):
        raise ValueError(
            "The length of target cell type and target gene should be the same."
        )

    pert_target_cell_type, pert_target_gene = (
        args.pert_target_cell_type,
        args.pert_target_gene,
    )
    if args.pert_target_gene_file:
        genes = []
        with open(args.pert_target_gene_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    genes.append(line)

        pert_target_cell_type = [pert_target_cell_type[0]] * len(genes)
        pert_target_gene = genes

    for target_cell_type, target_gene in zip(pert_target_cell_type, pert_target_gene):
        wandb_args = copy.deepcopy(args.wandb)
        trainer_args = copy.deepcopy(args.trainer)

        # modify wandb args and output dir
        wandb_args.name = f"{target_cell_type}_{target_gene}_{wandb_args.name}"
        trainer_args.output_folder = (
            f"{trainer_args.output_folder}/{target_cell_type}_{target_gene}"
        )

        init_wandb(wandb_args, asdict(args))

        target_gene_ind = gene_list.index(target_gene)
        task = PerturbationTask(
            target_cell_type=target_cell_type,
            target_gene=target_gene,
            target_gene_ind=target_gene_ind,
        )

        pert_model: PertModule = dispatch_pert_module(
            pert_type=args.pert.model_type,
            pert_num=args.pert.perturbation_num,
            pert_space=args.model.atac_model.input_dim,
            pert_val=pert_val,
            pert_step=args.trainer.pert_num_steps,
            pert_config=args.pert,
            embedding_module=model.atac_model.embedding,
        )

        head_model = ValueIndexerHead(
            target_gene_ind,
            increase_value=(
                False if args.trainer.rna_expression_direction == "down" else True
            ),
        )

        # init model and start perturbating
        trainer_cls = trainer_method[args.trainer_method]
        trainer = trainer_cls(
            pert_model,
            model,
            head_model,
            task,
            args=trainer_args,
            has_wandb_writer=True,
        )

        trainer.train()
        wandb.finish()


if __name__ == "__main__":
    main()
