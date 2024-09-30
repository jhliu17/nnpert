from typing import List
import numpy as np
import sklearn


def build_mapping_neighbors(
    mapping_array: np.ndarray,
    neighbor_array: np.ndarray,
    cell_type_label: np.ndarray,
    random_state: int,
):
    """Build the learned mapping model and the pretrained knn model

    :param mapping_array: data used for learning the mapping
    :param neighbor_array: data used for learning the knn
    :param cell_type_label: label used for learning the knn
    :param random_state: random state used for PCA
    :return: learned mapping model and knn model
    """
    # initialize cell type evaluator
    pca = sklearn.decomposition.PCA(n_components=30, random_state=random_state)
    pca.fit(mapping_array)

    neighbor_pca = pca.transform(neighbor_array)
    cell_type_cls = sklearn.neighbors.KNeighborsClassifier(n_neighbors=30)
    cell_type_cls.fit(neighbor_pca, cell_type_label)
    return pca, cell_type_cls


def eval_cell_type_ratio(
    input_array: np.ndarray,
    cell_type_label: np.ndarray,
    query_array: np.ndarray,
    ratio_keys: List[str],
    random_state: int,
):
    # initialize cell type evaluator
    pca, cell_type_cls = build_mapping_neighbors(
        input_array, input_array, cell_type_label, random_state=random_state
    )

    query_num = query_array.shape[0]
    query_cell_type = cell_type_cls.predict(pca.transform(query_array))
    ratio_results = {}
    for k in ratio_keys:
        ratio = np.sum(query_cell_type == k) / query_num
        ratio_results[k] = ratio

    return ratio_results
