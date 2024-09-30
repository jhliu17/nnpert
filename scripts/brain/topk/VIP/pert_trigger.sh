DATETIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_FOLDER=outputs/experiment_aurora/pert/brain_topk/pert_trigger_$DATETIME

# attack_target_cell=VIP VIP SST SST OPC OPC OPC Endo Endo Astro Oligo Oligo Micro Micro Astro Astro
# attack_target_gene=GAD1 GAD2 GAD1 GAD2 NXPH1 OLIG1 OLIG2 FLT1 CLDN5 ALDH1A1 MOBP MOG CX3CR1 APBB1IP AQP4 GJA1

export WANDB_API_KEY=062f6f4523d3b13835ab0f0e45f6bbb89d038e9f
python perturb_brain.py \
    --pert-target-cell-type VIP\
    --pert-target-gene Dummy \
    --pert-target-gene-file datasets/aurora/topk_expressed_genes/2023-07-31-20-35-40/VIP/VIP_topk_gene_names.txt \
    --pretrained-trainer-ckpt-path outputs/experiment_aurora/train/brain/cross_mapping_20240129_161848/model-final.pt \
    --model.atac-model.input-dim 127219 \
    --model.atac-model.chromosome-dims 11348 6169 6997 6364 3554 4233 4076 4178 4966 3189 3807 10272 3469 1478 2488 8367 6290 7655 7203 6950 6250 5547 2245 124 \
    --model.atac-model.hidden-dims 16 32\
    --model.atac-model.latent-dim 20 \
    --model.rna-model.input-dim 3000 \
    --model.rna-model.hidden-dims 640 320 \
    --model.rna-model.latent-dim 20 \
    --model.atac2rna-model.input-dim 20 \
    --model.atac2rna-model.latent-dim 128 \
    --model.atac2rna-model.affine-num 9 \
    --model.rna2atac-model.input-dim 20 \
    --model.rna2atac-model.latent-dim 128 \
    --model.rna2atac-model.affine-num 9 \
    --model.atac-discriminator.input-dim 20 \
    --model.atac-discriminator.affine-num 9 \
    --model.rna-discriminator.input-dim 20 \
    --model.rna-discriminator.affine-num 9 \
    --pert.model-type trigger \
    --pert.perturbation-num 10 \
    --pert.trigger.trigger-pert-type embedding_trigger_perturbation \
    --pert.trigger.replace-num-candidates 32 \
    --pert.trigger.use-optim-eval \
    --trainer.dataset-name brain \
    --trainer.output-folder $OUTPUT_FOLDER \
    --trainer.atac-seq-path datasets/aurora/brain/train_atac.npz \
    --trainer.atac-seq-label-path datasets/aurora/brain/atac_gene_activity_labels.txt \
    --trainer.rna-seq-path datasets/aurora/brain/train_rna.npz \
    --trainer.rna-seq-label-path datasets/aurora/brain/labels.txt \
    --trainer.train-atac-ae-lr 1e-3 \
    --trainer.train-rna-ae-lr 5e-4 \
    --trainer.train-ae-lr 5e-4 \
    --trainer.train-aff-lr 5e-4 \
    --trainer.train-gen-lr 5e-4 \
    --trainer.train-dis-lr 1e-4 \
    --trainer.train-batch-size 64 \
    --trainer.train-atac-ae-num-steps 36000 \
    --trainer.train-rna-ae-num-steps 9000 \
    --trainer.train-affine-num-steps 18000 \
    --trainer.update-dis-freq 5 \
    --trainer.lamda1 1 \
    --trainer.lamda2 5 \
    --trainer.lamda3 0.5 \
    --trainer.lamda4 0.5 \
    --trainer.focal-gamma 2 \
    --trainer.num-workers 4 \
    --trainer.gene-region-file datasets/aurora/brain/gene.regions.hg38.bed \
    --trainer.atac-description-file datasets/aurora/brain/atac_peaks.bed \
    --trainer.rna-description-file datasets/aurora/brain/gene_list.txt \
    --trainer.pert-num-steps 500 \
    --trainer.save-and-sample-every 100 \
    --trainer.pert-state-step -1 \
    --trainer.eval-trial-size 16 \
    --trainer.rna-expression-direction down \
    --trainer.atac-expression-direction down \
    --trainer.no-complement-target-cell-type \
    --slurm.mode slurm \
    --slurm.slurm-output-folder $OUTPUT_FOLDER/slurm \
    --slurm.cpus-per-task 6 \
    --slurm.node_list YOUR_NODE \
    --wandb.project aurora_topk \
    --wandb.name pert_trigger_$DATETIME \
    --wandb.notes ""