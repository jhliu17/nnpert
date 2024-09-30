import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=60) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(self.layer1(x))


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=90) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.q1, self.k1, self.v1 = [
            nn.Parameter(torch.rand([hidden_dim], requires_grad=True)) for _ in range(3)
        ]

    def forward(self, x):
        x = self.layer1(x)
        att_map = F.softmax((x * self.q1).T @ (x * self.k1), dim=-1)
        mid_out = (x * self.v1) @ att_map
        return self.layer2(mid_out)
