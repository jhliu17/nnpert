<h1 align="center"><strong>GRIDS: Understanding Transcriptional Regulatory Redundancies through Global Feature Explanations with Learnable Subset Perturbations</strong></h1>

Transcriptional regulation through cis-regulatory elements (CREs) is crucial for numerous biological functions, with its disruption potentially leading to various diseases. These CREs often exhibit redundancy, allowing them to compensate for each other in response to external disturbances, highlighting the need for methods to identify CRE sets that collaboratively regulate gene expression effectively. To address this, we introduce GRIDS, a model that approaches the task as a global feature explanation challenge to dissect combinatory CRE effects in two phases. First, GRIDS constructs a differentiable surrogate function to mirror the complex gene regulatory process, facilitating cross-translations in single-cell modalities. It then employs learnable perturbations within a state transition framework to offer global explanations, efficiently navigating the combinatorial feature landscape. Through comprehensive benchmarks in image classification and single-cell multi-omics, GRIDS demonstrates superior explanatory capabilities compared to other leading methods. Moreover, GRIDS's global explanations reveal intricate regulatory redundancy across cell types and states, underscoring its potential to advance our understanding of cellular regulation.


## Dependencies

1. Python: 3.10.12

2. Install other packages
    ```bash
    pip install -r requirements.txt
    ```

## Surrogate Model Training

We use `wandb` to visualize the training process and `slurm` to submit the job.
To see argument details, use `python train.py -h`. This process would run on Slurm.

Start training with GPU:
```bash
sh scripts/brain/train.sh
```

## Perturbation

Run perturbation on the pretrained surrogate model. This process would run on Slurm.
```bash
sh scripts/brain/pert_trigger.sh
```
