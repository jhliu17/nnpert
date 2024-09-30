DATETIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_FOLDER=outputs/experiment_aurora/pert/mnist/pert_cxplain_$DATETIME

export WANDB_API_KEY=062f6f4523d3b13835ab0f0e45f6bbb89d038e9f
python perturb_mnist.py \
    --trainer-method cxplain \
    --pert.model-type cxplain \
    --pert.perturbation-num 64 \
    --pert.cxplain.trigger-pert-type trigger_perturbation \
    --pert.cxplain.mask-type image \
    --pert.cxplain.batch-size 128 \
    --pert.cxplain.trainer-config.model-type unet \
    --pert.cxplain.trainer-config.batch-size 128 \
    --pert.cxplain.trainer-config.epoch 500 \
    --pert.cxplain.trainer-config.early-stopping-patience 10 \
    --pert.cxplain.trainer-config.learning-rate 5e-4 \
    --pert.cxplain.trainer-config.downsample-factors 2 \
    --pert.cxplain.trainer-config.num-units 1 \
    --trainer.data-folder datasets/mnist \
    --trainer.starting-num-type 8 \
    --trainer.pert-num-steps 500 \
    --trainer.pert-state-step -1 \
    --trainer.save-and-sample-every 10 \
    --trainer.train-batch-size 64 \
    --trainer.eval-batch-size 64 \
    --trainer.num-workers 2 \
    --trainer.output-folder $OUTPUT_FOLDER \
    --head-trainer.data-folder datasets/mnist \
    --head-trainer.output-folder $OUTPUT_FOLDER/head_model \
    --head-trainer.train-num-steps 2000 \
    --head-trainer.save-and-eval-every 500 \
    --head-trainer.num-type-list 8 3 \
    --head-trainer.tgt-num-type 8 \
    --slurm.mode slurm \
    --slurm.slurm-output-folder $OUTPUT_FOLDER/slurm \
    --slurm.cpus-per-task 4 \
    --slurm.node_list YOUR_NODE \
    --wandb.project aurora \
    --wandb.name pert_cxplain_mnist_$DATETIME \
    --wandb.notes "using train test splits"