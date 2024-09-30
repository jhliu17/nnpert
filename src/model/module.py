import math
import torch
import torch.nn as nn

from .op import look_up_category_embedding


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class CategoryEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, category_num: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.category_num = category_num
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.zeros((num_embeddings, category_num, embedding_dim))
        )
        nn.init.normal_(self.weight)

    def forward(self, x) -> torch.Tensor:
        """embedding for categorical features

        :param x: [batch, feat_num]
        """
        embed = look_up_category_embedding(x, self.weight)
        return embed


# Define the lambda layer
class LambdaLayer(nn.Module):
    def __init__(self, lambda_function):
        super(LambdaLayer, self).__init__()
        self.lambda_function = lambda_function

    def forward(self, x):
        return self.lambda_function(x)
