import torch
import scanpy as sc

import numpy as np
import anndata as ad
import pegasus as pg

from torch.utils.data import Dataset, Subset
from scipy.sparse import load_npz
from anndata import AnnData
from pegasusio import MultimodalData
from sklearn.preprocessing import LabelEncoder


def load_raw_dataset(
    dataset_name: str = None, dataset_path: str = None, read_cache: bool = False
):
    adata: ad.AnnData
    adata_raw: ad.AnnData

    if not read_cache:
        raise Exception("Please preprocess data...")
        # if dataset_name == "pancreas":
        #     adata = scv.datasets.pancreas()
        #     adata_raw = adata.copy()
        # else:
        #     raise Exception(f"Cannot found dataset {dataset_name}")

        # # preprocess dataset
        # adata, adata_raw = raw_dataset_processing(adata, adata_raw)
        # torch.save({"adata": adata, "adata_raw": adata_raw}, dataset_path)
    else:
        # read processed dataset
        # pth_dict = torch.load(dataset_path)
        # adata, adata_raw = pth_dict["adata"], pth_dict["adata_raw"]
        adata = sc.read_h5ad(dataset_path)
        adata_raw = sc.read_h5ad(f"{dataset_path}.raw")
    return adata, adata_raw


def deepvelo_dataset_processing(adata: ad.AnnData, adata_raw: ad.AnnData):
    X = np.tile(adata_raw.X.A[:, adata.var["velocity_genes"]], (5, 1))
    Y = np.tile(adata.layers["velocity"][:, adata.var["velocity_genes"]], (5, 1))
    noise_param = adata_raw.X.A.mean() / 20
    X[adata_raw.shape[0] :, :] += (
        X[adata_raw.shape[0] :, :] == 0
    ) * np.random.exponential(noise_param, X[adata_raw.shape[0] :, :].shape)
    return X, Y


def load_deepvelo_dataset(
    raw_dataset_path: str = None, dataset_path: str = None, read_cache: bool = False
):
    X: np.ndarray
    Y: np.ndarray

    if not read_cache:
        # read processed raw dataset
        adata: ad.AnnData
        adata_raw: ad.AnnData

        # pth_dict = torch.load(raw_dataset_path)
        # adata, adata_raw = pth_dict["adata"], pth_dict["adata_raw"]
        adata = sc.read_h5ad(raw_dataset_path)
        adata_raw = sc.read_h5ad(f"{raw_dataset_path}.raw")

        X, Y = deepvelo_dataset_processing(adata, adata_raw)
        torch.save({"X": X, "Y": Y}, dataset_path)
    else:
        # read processed dataset
        pth_dict = torch.load(dataset_path)
        X, Y = pth_dict["X"], pth_dict["Y"]

    return X, Y


class GeneTrajDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        starting_cell_type: str = None,
        global_cell_type_array: np.ndarray = None,
    ) -> None:
        super().__init__()
        self.npz = np.load(data_folder, allow_pickle=True)

        # use starting cell type to filter out cells if mask is None
        mask = None
        if starting_cell_type is not None:
            # subset npz should have the cell_type key, otherwise will use global cell
            # type array to filter out cells
            if "cell_type" in self.npz.keys():
                mask = self.npz["cell_type"] == starting_cell_type
            else:
                mask = global_cell_type_array == starting_cell_type

        self.traj = np.log1p(
            self.npz["path"].astype(np.float32)
            if mask is None
            else self.npz["path"].astype(np.float32)[:, mask]
        )

        self.time = self.npz["step"].astype(np.float32)

    def __len__(self):
        return self.traj.shape[1]

    def __getitem__(self, index):
        x = torch.from_numpy(self.traj[:, index])  # [t, sample, h] -> [t, h]
        return x


class CellTypeDataSet(Dataset):
    def __init__(self, genes, cell_types, d: dict) -> None:
        super().__init__()
        self.genes = genes
        self.cell_types = cell_types
        self.d = d

    def __len__(self):
        return self.genes.shape[0]

    def __getitem__(self, index):
        return self.genes[index], self.d[self.cell_types[index]]


class MNISTSubset(Dataset):
    def __init__(self, dataset, indices, label_mapping: dict, transform=None) -> None:
        super().__init__()
        self.subset = Subset(dataset, indices)
        self.label_mapping = label_mapping
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.subset[index]
        return self.transform(img), self.label_mapping[str(label)]

    def __len__(self):
        return len(self.subset)


class CountLogNormalizer(object):
    def __init__(self, norm_count: int = 1e5, **kwargs) -> None:
        self.norm_count = norm_count
        self.kwargs = kwargs

    def __call__(self, array_data: np.ndarray) -> np.ndarray:
        ann_data = AnnData(array_data)
        mul_data = MultimodalData(ann_data, modality="rna")
        pg.log_norm(mul_data, **self.kwargs)
        normed_data = mul_data.X.toarray()
        return normed_data


class SequenceDataSet(Dataset):
    def __init__(
        self,
        file_mat: str,
        file_cls: list[str],
        cls_encoder: LabelEncoder,
        do_normalization: bool = False,
        target_cell_type: str = None,
        complement_target_cell_type: bool = False,
        **normalizer_kwargs,
    ):
        self.file_mat = load_npz(file_mat)
        self.cls = np.array(cls_encoder.transform(file_cls), dtype=np.int32)

        self.normalizer = (
            CountLogNormalizer(**normalizer_kwargs) if do_normalization else None
        )
        if self.normalizer is None:
            self.dense_mat = self.file_mat.toarray().astype(np.float32)
            self.depths = np.sum(self.file_mat, axis=1).astype(np.float32)
        else:
            unormalized_dense_array = self.file_mat.toarray().astype(np.float32)
            self.dense_mat = self.normalizer(unormalized_dense_array)
            self.depths = np.sum(self.file_mat, axis=1).astype(np.float32)

        if target_cell_type is not None:
            target_mask = self.cls == cls_encoder.transform([target_cell_type])[0]

            if complement_target_cell_type:
                target_mask = np.logical_not(target_mask)

            self.dense_mat = self.dense_mat[target_mask]
            self.depths = self.depths[target_mask]
            self.cls = self.cls[target_mask]

    def __getitem__(self, index):
        feats = torch.from_numpy(self.dense_mat[index]).flatten()
        cls = int(self.cls[index])
        depths = torch.from_numpy(self.depths[index]).flatten()
        return feats, cls, depths

    def __len__(self):
        return self.dense_mat.shape[0]
