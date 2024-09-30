import numpy as np
import umap

from itertools import cycle
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder


def umap_transform(
    embedding: np.ndarray, umap_reducer: umap.UMAP = None, umap_config: dict = None
):
    if (umap_reducer is None and umap_config is None) or (
        umap_reducer is not None and umap_config is not None
    ):
        raise Exception("Should provide either `umap_reducer` or `umap_config`.")

    umap_embedding: np.ndarray = None
    if umap_reducer:
        umap_embedding = umap_reducer.transform(embedding)

    if umap_config:
        umap_reducer = umap.UMAP(**umap_config)
        umap_embedding = umap_reducer.fit_transform(embedding)

    return umap_embedding


def eval_embedding_scores(embedding: np.ndarray, label_array: np.ndarray):
    silhouette = silhouette_score(embedding, label_array)

    score = {"silhouette": silhouette}
    return score


def plot_single_embedding_umap(
    embedding_name: str, embedding: np.ndarray, labels: list[str], umap_config: dict
):
    umap_embedded = umap_transform(embedding, umap_config=umap_config)

    # prepare label array
    label_encoder = LabelEncoder()
    label_array = label_encoder.fit_transform(labels)

    # compute score
    silhouette = eval_embedding_scores(umap_embedded, label_array)["silhouette"]

    # plot umap
    fig, axes = plt.subplots()
    for i, c in enumerate(np.unique(labels)):
        mask = label_array == label_encoder.transform([c])[0]
        feature = umap_embedded[mask]
        axes.scatter(feature[:, 0], feature[:, 1], label=c, s=0.7, marker=".")
    axes.legend(markerscale=4, fontsize=6)
    axes.set_title(
        f"UMAP of {embedding_name.upper()} Embedding (silhouette={silhouette:.3f})"
    )
    output = {
        "plot": (fig, axes),
        "umap_embedding": umap_embedded,
        "silhouette_score": silhouette,
    }
    return output


def plot_joint_embedding_umap(
    embedding_name: str,
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    labels: list[str],
    umap_config: dict,
):
    embedding = np.concatenate((embedding1, embedding2), axis=0)
    umap_embedded = umap_transform(embedding, umap_config=umap_config)
    umap_embedded1, umap_embedded2 = np.split(umap_embedded, 2)

    # prepare label array
    label_encoder = LabelEncoder()
    label_array = label_encoder.fit_transform(labels)

    # compute score
    silhouette = eval_embedding_scores(umap_embedded, np.tile(label_array, 2))[
        "silhouette"
    ]

    # color
    colors = cycle(
        [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    )

    # plot umap
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, c in enumerate(np.unique(labels)):
        mask = label_array == label_encoder.transform([c])[0]
        color = next(colors)

        feature1 = umap_embedded1[mask]
        axes[0].scatter(
            feature1[:, 0],
            feature1[:, 1],
            label=c,
            s=0.7,
            marker=".",
            c=color,
            alpha=0.8,
        )
        axes[1].scatter(
            feature1[:, 0],
            feature1[:, 1],
            label=c,
            s=0.7,
            marker=".",
            c=color,
            alpha=0.8,
        )

        feature2 = umap_embedded2[mask]
        axes[0].scatter(
            feature2[:, 0],
            feature2[:, 1],
            label=f"{c} Cross",
            s=0.5,
            marker="v",
            c=color,
            alpha=0.8,
        )
        axes[2].scatter(
            feature2[:, 0],
            feature2[:, 1],
            label=f"{c} Cross",
            s=0.5,
            marker="v",
            c=color,
            alpha=0.8,
        )
    axes[0].legend(markerscale=4, fontsize=6)
    axes[0].set_title(
        f"UMAP of {embedding_name.upper()} Embedding (silhouette={silhouette:.3f})"
    )
    output = {
        "plot": (fig, axes),
        "umap_embedding": umap_embedded,
        "silhouette_score": silhouette,
    }
    return output
