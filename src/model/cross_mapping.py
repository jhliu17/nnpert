import torch
import torch.nn as nn

from dataclasses import dataclass, field
from .ae import ATACSeqAEModel, RNASeqAEModel
from .affine import AffineTransform, NaiveAffineDiscriminator


@dataclass
class AtacRnaCrossMappingConfig:
    atac_model: ATACSeqAEModel = field(default_factory=ATACSeqAEModel)
    rna_model: RNASeqAEModel = field(default_factory=RNASeqAEModel)
    atac2rna_model: AffineTransform = field(default_factory=AffineTransform)
    rna2atac_model: AffineTransform = field(default_factory=AffineTransform)
    atac_discriminator: NaiveAffineDiscriminator = field(
        default_factory=NaiveAffineDiscriminator
    )
    rna_discriminator: NaiveAffineDiscriminator = field(
        default_factory=NaiveAffineDiscriminator
    )


class AtacRnaCrossMappingModel(nn.Module):
    def __init__(self, config: AtacRnaCrossMappingConfig) -> None:
        super().__init__()

        self.atac_model = config.atac_model
        self.rna_model = config.rna_model
        self.atac2rna_model = config.atac2rna_model
        self.rna2atac_model = config.rna2atac_model
        self.atac_discriminator = config.atac_discriminator
        self.rna_discriminator = config.rna_discriminator

    @property
    def autoencoders(self):
        return [self.atac_model, self.rna_model]

    @property
    def affine_transforms(self):
        return [self.atac2rna_model, self.rna2atac_model]

    @property
    def generators(self):
        return [
            self.atac_model,
            self.rna_model,
            self.atac2rna_model,
            self.rna2atac_model,
        ]

    @property
    def discriminators(self):
        return [self.atac_discriminator, self.rna_discriminator]

    def encode_given_embed(self, x: torch.Tensor, mode: str):
        if mode == "atac":
            return self.atac_model.encode_given_embed(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def encode(self, x: torch.Tensor, mode: str):
        if mode == "atac":
            return self.atac_model.encode(x)
        elif mode == "rna":
            return self.rna_model.encode(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def decode(self, h: torch.Tensor, mode: str):
        if mode == "atac":
            return self.atac_model.decode(h)
        elif mode == "rna":
            return self.rna_model.decode(h)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def transform(self, x: torch.Tensor, mode: str):
        if mode == "atac2rna":
            return self.atac2rna_model(x)
        elif mode == "rna2atac":
            return self.rna2atac_model(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(
        self,
        x: torch.Tensor,
        affine_index: torch.Tensor = None,
        mode: str = "atac",
    ):
        if mode == "atac":
            return self.atac_model(x)
        elif mode == "rna":
            return self.rna_model(x)
        elif mode == "atac2rna":
            return self.decode(
                self.transform(self.encode(x, "atac"), "atac2rna"), "rna"
            )
        elif mode == "rna2atac":
            return self.decode(
                self.transform(self.encode(x, "rna"), "rna2atac"), "atac"
            )
        elif mode == "atac_dis":
            return self.atac_discriminator(x, affine_index)
        elif mode == "rna_dis":
            return self.rna_discriminator(x, affine_index)
        else:
            raise ValueError(f"Unknown mode: {mode}")
