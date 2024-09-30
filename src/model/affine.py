import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass(eq=False)
class AffineTransform(nn.Module):
    input_dim: int
    latent_dim: int
    affine_num: int
    affine_layer_num: int = 3

    def __post_init__(self) -> None:
        super().__init__()

        # affine matrix init
        self.affine_matrices = nn.ParameterList(
            [
                nn.Parameter(
                    torch.stack(
                        [
                            torch.randn(self.input_dim, self.input_dim).flatten()
                            for _ in range(self.affine_num)
                        ]
                    )
                )
                for _ in range(self.affine_layer_num)
            ]
        )
        self.affine_offsets = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.affine_num, self.input_dim))
                for _ in range(self.affine_layer_num)
            ]
        )

        # regressor for the affine transform selection
        self.fc_loc = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            nn.ReLU(True),
            nn.Linear(self.latent_dim, self.affine_num),
        )
        self.act = nn.PReLU()

    def forward(self, x):
        soft_idx = F.softmax(self.fc_loc(x), dim=-1)

        output = x.unsqueeze(-1)
        for i in range(self.affine_layer_num):
            affine_matrix = torch.mm(soft_idx, self.affine_matrices[i])
            affine_matrix = affine_matrix.view(
                -1, self.input_dim, self.input_dim
            )  # [b, d, d]
            affine_offset = torch.mm(soft_idx, self.affine_offsets[i])
            affine_offset = affine_offset.unsqueeze(-1)  # [b, d, 1]

            # do affine transform
            output = torch.bmm(affine_matrix, output) + affine_offset

            # do activation until output
            if i < self.affine_layer_num - 1:
                output = self.act(output)

        output = output.squeeze(-1)
        return output


@dataclass(eq=False)
class NaiveAffineDiscriminator(nn.Module):
    input_dim: int
    affine_num: int
    dropout: float = 0.1

    def __post_init__(self) -> None:
        super().__init__()
        self.w_d = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.PReLU(),
            nn.Linear(self.input_dim // 2, self.affine_num),
        )

    def forward(self, x, affine_index=None):
        score = self.w_d(x)

        if affine_index is None:
            return score
        else:
            if isinstance(affine_index, torch.Tensor):
                index = affine_index
            else:
                index = torch.tensor(affine_index, device=x.device)  # [batch_sz, ]
            affine_score = torch.gather(score, dim=-1, index=index.unsqueeze(-1))
            return affine_score
