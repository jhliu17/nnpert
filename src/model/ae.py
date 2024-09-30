import torch
import torch.nn as nn
from dataclasses import dataclass, field

from .config import ModelConfig
from .module import CategoryEmbedding, LambdaLayer


@dataclass
class AEModelConfig(ModelConfig):
    # input dim
    input_dim: int = 1024

    # output dim
    output_dim: int = 1024


class AEModel(nn.Module):
    def __init__(self, config: AEModelConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.output_layer = nn.Linear(64, config.output_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        h = self.decoder(h)
        dx = self.output_layer(h)
        return dx

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        dx = self.decode(h)
        return dx

    def get_l1_regularization(self):
        all_params = []
        for params in self.encoder.parameters():
            all_params.append(params.view(-1))
        for params in self.decoder.parameters():
            all_params.append(params.view(-1))

        l1_regularization = torch.norm(torch.cat(all_params), 1)
        return l1_regularization


class ShallowAEModel(nn.Module):
    def __init__(self, config: AEModelConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
        )

        self.output_layer = nn.Linear(64, config.output_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        h = self.decoder(h)
        dx = self.output_layer(h)
        return dx

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        dx = self.decode(h)
        return dx


@dataclass(eq=False)
class RNASeqAEModel(nn.Module):
    input_dim: int
    hidden_dims: list[int] = field(default_factory=lambda: [640, 320], hash=False)
    latent_dim: int = 20

    def __post_init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dims[0]),
            nn.BatchNorm1d(self.hidden_dims[0]),
            nn.PReLU(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.BatchNorm1d(self.hidden_dims[1]),
            nn.PReLU(),
            nn.Linear(self.hidden_dims[1], self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dims[1]),
            nn.BatchNorm1d(self.hidden_dims[1]),
            nn.PReLU(),
            nn.Linear(self.hidden_dims[1], self.hidden_dims[0]),
            nn.BatchNorm1d(self.hidden_dims[0]),
            nn.PReLU(),
            nn.Linear(self.hidden_dims[0], self.input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        dx = self.decoder(h)
        return dx

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        dx = self.decode(h)
        return dx


@dataclass(eq=False)
class ATACSeqAEModel(nn.Module):
    input_dim: int
    chromosome_dims: list[int] = field(default_factory=list, hash=False)
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32], hash=False)
    latent_dim: int = 20

    def __post_init__(self) -> None:
        super().__init__()
        self.embedding = CategoryEmbedding(
            num_embeddings=self.input_dim,
            category_num=2,
            embedding_dim=self.hidden_dims[0],
        )

        self.in_split_layer = nn.ModuleList()
        for _ in range(len(self.chromosome_dims)):
            self.in_split_layer.append(
                nn.Sequential(
                    LambdaLayer(lambda x: x.sum(dim=1)),
                    nn.BatchNorm1d(self.hidden_dims[0]),
                    nn.PReLU(),
                    nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
                    nn.BatchNorm1d(self.hidden_dims[1]),
                    nn.PReLU(),
                )
            )

        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_dims[1] * len(self.chromosome_dims), self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.out_split_layer = nn.ModuleList()
        for n in self.chromosome_dims:
            self.out_split_layer.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dims[1], self.hidden_dims[0]),
                    nn.BatchNorm1d(self.hidden_dims[0]),
                    nn.PReLU(),
                    nn.Linear(self.hidden_dims[0], n),
                )
            )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dims[1] * len(self.chromosome_dims)),
            nn.BatchNorm1d(self.hidden_dims[1] * len(self.chromosome_dims)),
        )

    def encode_given_embed(self, x: torch.Tensor) -> torch.Tensor:
        xs = torch.split(x, self.chromosome_dims, dim=1)
        enc_chroms = []
        for layer, chrom_input in zip(self.in_split_layer, xs):
            enc_chroms.append(layer(chrom_input))

        x = torch.cat(enc_chroms, dim=1)
        h = self.encoder(x)
        return h

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        h = self.encode_given_embed(x)
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        h = self.decoder(h)
        xs = torch.chunk(h, chunks=len(self.chromosome_dims), dim=1)
        rec_chroms = []
        for layer, chrom_input in zip(self.out_split_layer, xs):
            rec_chroms.append(layer(chrom_input))

        dx = torch.cat(rec_chroms, dim=1)
        return dx

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        dx = self.decode(h)
        return dx
