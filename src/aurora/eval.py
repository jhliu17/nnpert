import numpy as np

from typing import Union
from .utils import (
    parse_gene_regions_bed_file,
    parse_peeks_bed_file,
    compute_range_distance,
    AtacBed,
)

ONEMB = 1e6


def get_target_gene_loc(gene_regions: AtacBed, target_gene):
    gene_mask = gene_regions.gene_name == target_gene
    genes_chr, genes_reg = (
        gene_regions.chromosome[gene_mask],
        gene_regions.peak_range[gene_mask],
    )
    return genes_chr, genes_reg


def get_search_loc(atac_peaks: AtacBed, search_ranges):
    search_chr, search_reg = (
        atac_peaks.chromosome[search_ranges],
        atac_peaks.peak_range[search_ranges],
    )
    return search_chr, search_reg


def compute_search_metrics(results: dict, soft_threshold, out_of_dist=1000 * ONEMB):
    metrics = {
        "hit": 1 if np.isclose(results["closest_distance"], 0) else 0,
        "soft_hit": 1
        if np.isclose(results["closest_distance"], 0, atol=soft_threshold)
        else 0,
        "soft_hit_num": np.sum(results["inchrom_distances"] < soft_threshold).item(),
        "soft_threshold_mb": soft_threshold / ONEMB,
        "closest_dist_mb": (results["closest_distance"] / ONEMB).item()
        if np.isfinite(results["closest_distance"])
        else float(out_of_dist / ONEMB),
        "inchr_num": len(results["inchrom_peaks"]),
        "search_num": results["search_num"],
        "inchr_ratio": len(results["inchrom_peaks"]) / results["search_num"],
    }

    return metrics


def eval_search_performance(
    gene_region_file: str,
    atac_peak_file: str,
    target_gene_name: str,
    search_peaks: Union[np.ndarray, list],
    soft_threshold: float = 1e7,
):
    """Parse the bed file, get the region and then compute distance

    :param gene_region_file: gene region bed file
    :param atac_peak_file: atac peak bed file
    :param target_gene_name: the searching target
    :param search_peaks: searching peaks
    """
    if isinstance(search_peaks, list):
        search_peaks = np.array(search_peaks)

    gene_regions = parse_gene_regions_bed_file(gene_region_file)
    atac_peaks = parse_peeks_bed_file(atac_peak_file)

    genes_chr, genes_reg = get_target_gene_loc(gene_regions, target_gene_name)
    search_chr, search_reg = get_search_loc(atac_peaks, search_peaks)

    dist = compute_range_distance(search_chr, search_reg, genes_chr, genes_reg)

    # post processing eval results
    closest_dist_id = dist[0].argmin()
    closest_dist = dist[0][closest_dist_id]
    closest_peak = search_peaks[closest_dist_id]
    inchr_dists_mask = np.isfinite(dist[0])
    inchr_dists = dist[0][inchr_dists_mask]
    inchr_peaks = search_peaks[inchr_dists_mask]
    results = {
        "closest_distance": closest_dist,
        "closest_peak": closest_peak,
        "inchrom_distances": inchr_dists,
        "inchrom_peaks": inchr_peaks.tolist(),
        "search_num": len(search_peaks),
    }

    metrics = compute_search_metrics(results, soft_threshold=soft_threshold)
    return metrics


if __name__ == "__main__":
    search_peaks = [16, 308]

    gene_region_file = "datasets/aurora/brain/gene.regions.hg38.bed"
    atac_peak_file = "datasets/aurora/brain/atac_peaks.bed"
    target_gene = "GAD1"
    results = eval_search_performance(
        gene_region_file, atac_peak_file, target_gene, search_peaks
    )
    print(results)
