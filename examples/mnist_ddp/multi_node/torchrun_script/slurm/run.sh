#!/bin/bash

#SBATCH --job-name=multinode-example
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=2

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo SLURM_JOB_LIST $SLURM_JOB_NODELIST
echo NODES $nodes
echo HEAD NODE $head_node
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun torchrun \
    --nnodes 2 \
    --nproc_per_node 2 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    /home/nick/GitRepos/torchface/examples/mnist_ddp/multi_node/main.py
