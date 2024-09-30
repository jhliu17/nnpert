import numpy as np
import matplotlib.pyplot as plt


def plot_gene_expr(
    gene_name: str,
    before_rna_exprs: np.ndarray,
    after_rna_exprs: np.ndarray,
    bins: int = 20,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[1].hist(
        before_rna_exprs,
        bins=bins,
        alpha=0.8,
        linewidth=0.5,
        edgecolor="white",
        label="Before",
    )
    axes[2].hist(
        after_rna_exprs,
        bins=bins,
        alpha=0.8,
        linewidth=0.5,
        edgecolor="white",
        label="After",
    )

    n_origin, *_ = axes[0].hist(
        before_rna_exprs,
        bins=bins,
        alpha=0.8,
        linewidth=0.5,
        edgecolor="white",
        label="Before",
    )
    n_attack, *_ = axes[0].hist(
        after_rna_exprs,
        bins=bins,
        alpha=0.8,
        linewidth=0.5,
        edgecolor="white",
        label="After",
    )

    min_val = min(np.min(before_rna_exprs), np.min(after_rna_exprs))
    max_val = max(np.max(before_rna_exprs), np.max(after_rna_exprs))
    max_n = np.max(np.concatenate((n_origin, n_attack)))
    for i in range(3):
        axes[i].set_xlim(min_val, max_val)
        axes[i].set_ylim(0, max_n + 5)
        axes[i].legend()
        axes[i].grid(True)
    axes[0].set_title(f"RNA Expression of {gene_name}")
    return fig, axes
