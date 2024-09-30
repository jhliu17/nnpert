import torch
import torch.nn as nn

from skorch import NeuralNet
from skorch.callbacks import EarlyStopping
from skorch.helper import predefined_split
from dataclasses import dataclass
from torch.utils.data import TensorDataset
from typing import Callable, Literal
from tqdm import tqdm
from ..utils.mask import MaskInput
from .module import MLPModelBuilder, UNetModelBuilder, BalanceKLDivLoss


@torch.inference_mode()
def create_dataset(
    samples: torch.Tensor,
    forward_fn: Callable,
    loss_fn: Callable,
    mask_fn: MaskInput,
    mask_value: torch.Tensor,
) -> TensorDataset:
    list_of_targets = []
    list_of_samples = torch.split(samples, 1, dim=0)
    for sample in tqdm(list_of_samples, desc="creating cxplain dataset"):
        list_of_importances = []
        for masked_samples in mask_fn.mask_sample(sample, mask_value):
            origin_loss = loss_fn(forward_fn(sample))
            masked_loss = loss_fn(forward_fn(masked_samples))
            importances = masked_loss - origin_loss
            list_of_importances.append(importances)

        importances = torch.cat(list_of_importances, dim=0)

        importances = torch.maximum(
            importances, importances.new_tensor(torch.finfo(importances.dtype).tiny)
        )

        normalized_importances = importances / torch.sum(
            importances, dim=0, keepdim=True
        )  # [N, 1]
        list_of_targets.append(normalized_importances.flatten())

    targets = torch.stack(list_of_targets, dim=0)
    dataset = TensorDataset(samples, targets)
    return dataset


@dataclass
class CXPlainTrainerConfig:
    model_type: Literal["mlp", "unet"] = "mlp"
    unet_conv_type: Literal["1d", "2d"] = "2d"
    batch_size: int = 128
    epoch: int = 500
    early_stopping_patience: int = 10
    learning_rate: float = 1e-3
    downsample_factors: int = 2
    num_layers: int = 2
    num_units: int = 64
    seed: int = 2024
    use_balance_kl_loss: bool = False


def train_cxplain(
    train_ds: TensorDataset,
    valid_ds: TensorDataset,
    device: torch.DeviceObjType,
    config: CXPlainTrainerConfig,
) -> torch.nn.Module:
    model_types = {
        "mlp": MLPModelBuilder,
        "unet": UNetModelBuilder,
    }

    early_stop_callback = EarlyStopping(
        patience=config.early_stopping_patience, threshold_mode="abs"
    )
    model = NeuralNet(
        module=model_types[config.model_type],
        module__downsample_factors=config.downsample_factors,
        module__num_layers=config.num_layers,
        module__num_units=config.num_units,
        module__conv_type=config.unet_conv_type,
        criterion=nn.KLDivLoss if not config.use_balance_kl_loss else BalanceKLDivLoss,
        criterion__reduction="batchmean",
        optimizer=torch.optim.Adam,
        optimizer__lr=config.learning_rate,
        train_split=predefined_split(valid_ds),
        batch_size=config.batch_size,
        max_epochs=config.epoch,
        device=device,
        callbacks=[early_stop_callback],
        iterator_train__shuffle=True,
    )

    model.fit(train_ds)
    torch_module = model.module_
    return torch_module
