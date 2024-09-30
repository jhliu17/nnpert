DATETIME=$(date +%Y%m%d_%H%M%S)
OUTPUT_FOLDER=outputs/experiment_aurora/pert/mnist/pert_trigger_$DATETIME

export WANDB_API_KEY=062f6f4523d3b13835ab0f0e45f6bbb89d038e9f
python perturb_mnist.py --pert.model-type trigger \
    --pert.perturbation-num 64 \
    --pert.trigger.replace-num-candidates 32 \
    --pert.trigger.use-optim-eval \
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
    --head-trainer.tgt-num-type 3 \
    --slurm.mode slurm \
    --slurm.slurm-output-folder $OUTPUT_FOLDER/slurm \
    --slurm.cpus-per-task 4 \
    --slurm.node_list YOUR_NODE \
    --wandb.project aurora \
    --wandb.name pert_trigger_mnist_$DATETIME \
    --wandb.notes "using train test splits"