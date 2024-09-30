import os
import torch
import numpy as np
import pandas as pd

from numpy.random import default_rng
from dataclasses import dataclass
from matplotlib import pyplot as plt


@dataclass
class AtacBed:
    chromosome: np.ndarray
    gene_name: np.ndarray
    peak_range: np.ndarray

    def get_gene_mask(self, gene: str):
        return self.gene_name == gene


def parse_gene_regions_bed_file(bed_file_path: str) -> AtacBed:
    data_frame = pd.read_csv(bed_file_path, delimiter="\t", header=None)
    return AtacBed(
        chromosome=data_frame.iloc[:, 0].to_numpy(),
        gene_name=data_frame.iloc[:, 3].to_numpy(),
        peak_range=np.sort(data_frame.iloc[:, 1:3].to_numpy(dtype=np.int64), axis=-1),
    )


def parse_peeks_bed_file(bed_file_path: str) -> AtacBed:
    data_frame = pd.read_csv(bed_file_path, delimiter="\t", header=None)
    return AtacBed(
        chromosome=data_frame.iloc[:, 0].to_numpy(),
        gene_name=None,
        peak_range=np.sort(data_frame.iloc[:, 1:3].to_numpy(dtype=np.int64), axis=-1),
    )


def plot_gene_expr(
    gene_name: str,
    origin_rna_exprs: np.ndarray,
    attack_rna_exprs: np.ndarray,
    bins: int = 20,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[1].hist(
        origin_rna_exprs,
        bins=bins,
        alpha=0.8,
        linewidth=0.5,
        edgecolor="white",
        label="Origin",
    )
    axes[2].hist(
        attack_rna_exprs,
        bins=bins,
        alpha=0.8,
        linewidth=0.5,
        edgecolor="white",
        label="Attack",
    )

    n_origin, *_ = axes[0].hist(
        origin_rna_exprs,
        bins=bins,
        alpha=0.8,
        linewidth=0.5,
        edgecolor="white",
        label="Origin",
    )
    n_attack, *_ = axes[0].hist(
        attack_rna_exprs,
        bins=bins,
        alpha=0.8,
        linewidth=0.5,
        edgecolor="white",
        label="Attack",
    )

    min_val, max_val = np.min(origin_rna_exprs), np.max(origin_rna_exprs)
    max_n = np.max(np.concatenate((n_origin, n_attack)))
    for i in range(3):
        axes[i].set_xlim(min_val, max_val)
        axes[i].set_ylim(0, max_n + 5)
        axes[i].legend()
        axes[i].grid(True)
    axes[0].set_title(f"RNA Expression of {gene_name}")
    return fig, axes


def range_distance(input: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute range distance between input and target,

    the distance is less than zero if there exists an overlap between the input rabge
    and the target range

    :param input: input range [N, 2]
    :param target: target range [M, 2]
    :return: pair-wise distance [N, M]
    """
    # first we sort the range (some ranges are in reverse order)
    input = np.sort(input, axis=-1)
    target = np.sort(target, axis=-1)

    start = np.maximum(input[:, None, 0], target[None, :, 0])
    end = np.minimum(input[:, None, 1], target[None, :, 1])
    distance = start - end

    # distance where the value less than zero is clamped to zero
    distance = np.clip(distance, 0, None)
    return distance


def compute_range_distance(
    input_chr: np.ndarray,
    input_range: np.ndarray,
    target_chr: np.ndarray,
    target_range: np.ndarray,
):
    """compute distatnces

    :param input_chr: [A, ] A's chromosome indexes
    :param input_range: [A, 2] A's chromosome ranges
    :param target_chr: [B,] B's chromosome indexes
    :param target_range: [B, 2] B's chromosome ranges
    """
    dist_list = []
    for i, gene_chr in enumerate(target_chr):
        dist = np.zeros_like(input_chr, dtype=np.float64)
        valid_range_mask = input_chr == gene_chr
        if np.any(valid_range_mask):
            valid_input_range = input_range[valid_range_mask]
            valid_dist = range_distance(valid_input_range, target_range[i : i + 1])
            dist[valid_range_mask] = valid_dist.flatten()

        # invalid ranges are marked with np.inf
        dist[np.logical_not(valid_range_mask)] = np.inf
        dist_list.append(dist)

    all_dist: np.ndarray = np.stack(dist_list, axis=-1)
    min_dist: np.ndarray = np.min(all_dist, axis=-1)
    return min_dist, all_dist


def parse_attack_str(s: str, separate_token=","):
    return s.split(separate_token)


def parse_topk_file(file_path: str):
    _, file_name = os.path.split(file_path)
    cell_type = file_name.split("_")[0]

    genes = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                genes.append(line)

    target_cells = [cell_type] * len(genes)
    target_genes = genes
    return target_cells, target_genes


def init_enhancer(
    input_dim: list[int],
    num_enhancer_tokens: int,
    initial_enhancers: str,
    invalid_enhancers: str,
    constraint_enhancers: str,
    rng,
):
    # init enhancers
    constraint_token_ids = [int(i) for i in constraint_enhancers.split(",") if i]
    invalid_token_ids = [int(i) for i in invalid_enhancers.split(",") if i]

    # to enable constraint token ids, just add its complementaries to the
    # invalid_token_ids
    if constraint_token_ids:
        complement_token_ids = [
            i for i in range(sum(input_dim)) if i not in constraint_token_ids
        ]
        combined_invalid_token_ids = invalid_token_ids + complement_token_ids
        invalid_token_ids = list(set(combined_invalid_token_ids))

    if initial_enhancers:
        # parse initial enhancers
        enhancer_token_ids = sorted([int(i) for i in initial_enhancers.split(",") if i])
    else:
        # otherwise randomly generate enhancers
        atac_list = [i for i in range(sum(input_dim)) if i not in invalid_token_ids]
        rng.shuffle(atac_list)
        enhancer_token_ids = sorted(atac_list[:num_enhancer_tokens])

    # check validity and length of enhancers
    assert len(enhancer_token_ids) == num_enhancer_tokens
    assert all([i not in invalid_token_ids for i in enhancer_token_ids])

    # set enhancer
    return enhancer_token_ids, invalid_token_ids


def determine_search_regions(
    input_dim: list[int],
    num_enhancer_tokens: int,
    initial_enhancers,
    invalid_enhancers,
    constraint_enhancers,
    rng,
    seen_enhancer_tokens: list[int] = None,
):
    _, invalid_token_ids = init_enhancer(
        input_dim,
        num_enhancer_tokens,
        initial_enhancers,
        invalid_enhancers,
        constraint_enhancers,
        rng,
    )

    set_of_invalid_token_ids = set(invalid_token_ids)
    if seen_enhancer_tokens is not None:
        set_of_invalid_token_ids.update(seen_enhancer_tokens)

    atac_list = [i for i in range(sum(input_dim)) if i not in set_of_invalid_token_ids]
    return atac_list


def device_setup(seed):
    # device setting
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # reproducible
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng = default_rng(seed)
    return use_cuda, device, rng


def inv_sigmoid(x: torch.Tensor, min_x=1e-5):
    max_x = 1 - min_x
    x = x.clamp(min_x, max_x)  # clamp for numerical stability
    y = torch.log(x / (1 - x))
    return y


def load_gene_list(path: str) -> list[str]:
    gene_list = []
    with open(path, "r") as f:
        for line in f:
            gene_name = line.strip()
            if gene_name:
                gene_list.append(gene_name)
    return gene_list


if __name__ == "__main__":
    input_chr = np.array([1, 1, 1])
    input_range = np.array([[0, 5], [4, 10], [9, 10]])
    target_chr = np.array([1])
    target_range = np.array([[0, 7]])

    print(compute_range_distance(input_chr, input_range, target_chr, target_range))
