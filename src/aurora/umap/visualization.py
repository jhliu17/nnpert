import numpy as np

from .utils import plot_single_embedding_umap, plot_joint_embedding_umap


def get_joint_umap_figure(
    atac2rna_latent: np.ndarray,
    atac2atac_latent: np.ndarray,
    rna2rna_latent: np.ndarray,
    rna2atac_latent: np.ndarray,
    labels,
    umap_config,
):
    rna_plot_results = plot_single_embedding_umap(
        "rna2rna", rna2rna_latent, labels, umap_config
    )
    atac_plot_results = plot_single_embedding_umap(
        "atac2atac", atac2atac_latent, labels, umap_config
    )
    rna_joint_plot_results = plot_joint_embedding_umap(
        "rna_joint",
        rna2rna_latent,
        atac2rna_latent,
        labels,
        umap_config,
    )
    atac_joint_plot_results = plot_joint_embedding_umap(
        "atac_joint",
        atac2atac_latent,
        rna2atac_latent,
        labels,
        umap_config,
    )
    figure_dict = {
        "rna2rna": rna_plot_results["plot"][0],
        "atac2atac": atac_plot_results["plot"][0],
        "rna_joint": rna_joint_plot_results["plot"][0],
        "atac_joint": atac_joint_plot_results["plot"][0],
    }
    score_dict = {
        "rna2rna_silhouette": rna_plot_results["silhouette_score"],
        "atac2atac_silhouette": atac_plot_results["silhouette_score"],
        "rna_joint_silhouette": rna_joint_plot_results["silhouette_score"],
        "atac_joint_silhouette": atac_joint_plot_results["silhouette_score"],
    }
    return figure_dict, score_dict


def get_single_umap_figure(
    rna2rna_latent: np.ndarray,
    atac2atac_latent: np.ndarray,
    labels,
    umap_config,
    source,
):
    if source == "rna":
        plot_results = plot_single_embedding_umap(
            "rna2rna", rna2rna_latent, labels, umap_config
        )
        figure_dict = {"rna2rna": plot_results["plot"][0]}
        score_dict = {"rna2rna_silhouette": plot_results["silhouette_score"]}
    elif source == "atac":
        plot_results = plot_single_embedding_umap(
            "atac2atac", atac2atac_latent, labels, umap_config
        )
        figure_dict = {"atac2atac": plot_results["plot"][0]}
        score_dict = {"atac2atac_silhouette": plot_results["silhouette_score"]}
    else:
        raise Exception(f"{source} cannot be understood.")

    return figure_dict, score_dict
