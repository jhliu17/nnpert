import os
import git
import wandb

from dataclasses import dataclass


@dataclass
class WandbConfig:
    # project name in wandb
    project: str = "grids"

    # wandb user name
    entity: str = ""

    # wandb run name
    name: str = ""

    # wandb run notes
    notes: str = ""

    # log git hash
    log_git_hash: bool = True

    # log code
    log_code: bool = True

    # code root
    code_root: str = "."


def init_wandb(args: WandbConfig, run_config: dict):
    if "WANDB_API_KEY" in os.environ["WANDB_API_KEY"]:
        wandb.login(key=os.environ["WANDB_API_KEY"])

    if args.log_git_hash:
        repo = git.Repo(search_parent_directories=True)
        git_hash = repo.head.object.hexsha
        args.notes = (
            f"{args.notes + ', ' if args.notes else ''}" + f"git hash: {git_hash}"
        )

    wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
        notes=args.notes,
        config=run_config,
    )
    wandb.run.log_code(
        root=args.code_root,
        include_fn=lambda path, root: path.endswith(".py") or path.endswith(".sh"),
    )
