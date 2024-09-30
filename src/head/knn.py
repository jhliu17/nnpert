from typing import Dict
import torch
import torch.nn as nn
import numpy as np

from anndata import AnnData
from .base import HeadModule
from ..eval import build_mapping_neighbors


class KNNHead(HeadModule):
    def __init__(self, k: int, adata: AnnData, tgt_cell_type: str):
        super().__init__()
        self.k = k

        # initialize targeting components
        (
            self.tgt_genes,
            self.tgt_pca,
            self.tgt_cls,
        ) = self._build_targeting_components(adata, tgt_cell_type)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        state = {
            "k": self.k,
            "tgt_genes": self.tgt_genes,
            "tgt_pca": self.tgt_pca,
            "tgt_cls": self.tgt_cls,
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.k = state_dict["k"]
        self.tgt_genes = state_dict["tgt_genes"]
        self.tgt_pca = state_dict["tgt_pca"]
        self.tgt_cls = state_dict["tgt_cls"]

    def _build_targeting_components(self, adata, tgt_cell_type):
        tgt_cell_sample_mask = (adata.obs.clusters == tgt_cell_type).to_numpy()
        raw_genes = adata.X.A[:, adata.var["velocity_genes"]]
        cell_types = adata.obs.clusters.to_numpy()
        tgt_genes = raw_genes[tgt_cell_sample_mask]
        pca, cell_type_cls = build_mapping_neighbors(
            raw_genes,
            tgt_genes,
            cell_types[tgt_cell_sample_mask],
            random_state=self.pert_args.seed,
        )
        return tgt_genes, pca, cell_type_cls

    def get_pert_target(self, x: torch.Tensor):
        """generate batch

        :param x: query tensor [batch, feat_num]
        """
        xnp = x.numpy(force=True)
        pca_xnp = self.tgt_pca.transform(xnp)
        _, nbind_xnp = self.tgt_cls.kneighbors(pca_xnp, n_neighbors=3)

        batch_num = nbind_xnp.shape[0]
        neigh_num = nbind_xnp.shape[-1]
        neigh_points = self.tgt_genes[nbind_xnp.flatten()].reshape(
            batch_num, neigh_num, -1
        )
        pseudo_target = np.mean(neigh_points, axis=1)
        y = x.new_tensor(pseudo_target)
        return y

    def get_loss(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        tgt_x = self.get_pert_target(x)
        loss_fn = nn.L1Loss(**kwargs)
        loss = loss_fn(x, tgt_x)
        return loss
