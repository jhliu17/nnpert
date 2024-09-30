import torch
import umap
import math
import scanpy as sc
import scvelo as scv
import numpy as np
import sklearn
import anndata as ad

from matplotlib import pyplot as plt


def load_dataset(path: str):
    adata = sc.read_h5ad(path)
    return adata


def exists(x):
    return x is not None


def divisible_by(numer, denom):
    return (numer % denom) == 0


def cycle_dataloader(dl):
    while True:
        for data in dl:
            yield data


@torch.inference_mode()
def sample_vector_field(vf_model, t: float, query_positions: np.ndarray, device):
    data = torch.from_numpy(query_positions).to(device)
    time = data.new_ones((data.size(0),)) * t
    vec = vf_model(time, data)
    return vec


def visualize_trajs(
    raw_gene_array: np.ndarray, raw_gene_color: np.ndarray, trajs: np.ndarray
):
    """generate grid plot to visualize different trajs

    :param raw_gene_array: origin gene matrix [cell, gene]
    :param raw_gene_color: origin gene color map [cell]
    :param trajs: trajectories to be visulized [time, cell, gene]
    """
    pca_dim = 30
    umap_neigh_num = 30

    # use PCA to reduce the dim of the count data
    pca = sklearn.decomposition.PCA(n_components=pca_dim, random_state=42)
    adata_pca = pca.fit_transform(raw_gene_array)

    # Further reduce the dim with UMAP
    umap_reducer = umap.UMAP(n_neighbors=umap_neigh_num, n_jobs=1, random_state=42)
    adata_umap = umap_reducer.fit_transform(adata_pca)

    # construct KNN with PCA
    neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=umap_neigh_num)
    neigh.fit(adata_pca)

    # reduction on trajs
    step_num, traj_num, gene_num = trajs.shape
    trajs_pca = pca.transform(trajs.reshape(-1, gene_num))
    trajs_umap = umap_reducer.transform(trajs_pca).reshape(-1, traj_num, 2)
    trajs_color = np.arange(step_num)

    row_num = int(math.sqrt(traj_num))
    fig, axes = plt.subplots(
        nrows=row_num, ncols=row_num, figsize=(2 * row_num, 1.5 * row_num)
    )
    for r in range(row_num):
        for c in range(row_num):
            offset = r * row_num + c
            if offset >= traj_num:
                break

            ax = axes[r][c]

            # base
            ax.scatter(
                adata_umap[:, 0],
                adata_umap[:, 1],
                s=1,
                alpha=0.5,
                c=raw_gene_color,
            )

            # trajectory
            ax.scatter(
                trajs_umap[:, offset, 0],
                trajs_umap[:, offset, 1],
                c=trajs_color,
                cmap="viridis",
                marker="*",
                s=20,
            )

            # starting point
            ax.scatter(
                trajs_umap[0, offset, 0],
                trajs_umap[0, offset, 1],
                s=20,
                color="red",
                marker="*",
            )

            ax.tick_params(axis="both", which="both", length=0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    return fig, axes


def visualize_vector_field(
    adata: ad.AnnData, velocity: np.ndarray, velocity_name: str = "neural_vf"
):
    pad_velocity = adata.layers["velocity"].copy()
    pad_velocity[:, adata.var["velocity_genes"]] = velocity
    adata.layers[velocity_name] = pad_velocity
    scv.tl.velocity_graph(adata, vkey=velocity_name)

    fig, ax = plt.subplots()
    scv.pl.velocity_embedding_stream(
        adata, basis="umap", vkey=velocity_name, recompute=True, ax=ax
    )
    return fig, ax


def create_pert_val(adata_path):
    adata = sc.read_h5ad(adata_path)
    raw_genes = adata.X.A[:, adata.var["velocity_genes"]]
    pert_val = np.percentile(raw_genes, 99, axis=0)
    return pert_val


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def group_mean(inp: torch.Tensor, group_num: int, group_size: int):
    """calculate group mean of inp

    :param inp: input tensor [group_num * group_size, feat]
    :param group_num: how many group
    :param group_size: how many group member for each group
    :return: group mean
    """
    if inp.dim() == 1:
        inp = inp.unsqueeze(-1)

    group_rows = torch.arange(group_num).repeat_interleave(group_size).to(inp.device)
    group_cols = torch.arange(group_num * group_size).to(inp.device)
    grouper = torch.zeros((group_num, group_num * group_size)).to(inp.device)

    grouper[group_rows, group_cols] = 1
    grouper = torch.nn.functional.normalize(grouper, p=1, dim=1)
    res = torch.mm(grouper, inp)
    return res


def read_and_post_process_adata(dataset_name: str, adata_path: str):
    """Read and do post-processing on adata

    :param adata_path: the path to the adata file (not adata raw file)
    """
    adata = sc.read_h5ad(adata_path)

    if dataset_name == "dentategyrus":

        def clusters_map_fn(x):
            if x in {"CA1-Sub", "CA2-3-4"}:
                return "CA"
            elif x in {"ImmGranule1", "ImmGranule2"}:
                return "Granule"
            else:
                return x

        adata.obs.clusters = adata.obs.clusters.map(clusters_map_fn).astype("category")
    else:
        pass

    return adata


def visualize_changed_mnist_digit(
    imgs: np.ndarray,
    labels: np.ndarray,
    pert_imgs: np.ndarray,
    pert_labels: np.ndarray,
    ind2num_type_dict: dict[int, str],
):
    """Visualize the changed MNIST digit

    :param imgs: original images
    :param labels: original labels
    :param perturbed_imgs: perturbed images
    :param perturbed_labels: perturbed labels
    """
    num = len(imgs)
    fig, axes = plt.subplots(nrows=4, ncols=max(2, num), figsize=(2 * num, 8))
    for i in range(num):
        axes[0][i].imshow(imgs[i].reshape(28, 28), cmap="gray")
        axes[0][i].set_title("pred: " + ind2num_type_dict[labels[i].item()])
        axes[0][i].axis("off")

        axes[1][i].imshow(
            (imgs[i] != pert_imgs[i]).astype(np.float32).reshape(28, 28), cmap="gray"
        )
        axes[1][i].set_title("pert loc")
        axes[1][i].axis("off")

        axes[2][i].imshow((imgs[i] - pert_imgs[i]).reshape(28, 28), cmap="gray")
        axes[2][i].set_title("pert diff")
        axes[2][i].axis("off")

        axes[3][i].imshow(pert_imgs[i].reshape(28, 28), cmap="gray")
        axes[3][i].set_title("pred: " + ind2num_type_dict[pert_labels[i].item()])
        axes[3][i].axis("off")
    return fig
