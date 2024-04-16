#! /bin/bash

torchrun \
    --standalone \
    --nproc_per_node=1 \
    $HOME/GitRepos/torchface/examples/mnist_ddp/single_node/main.py
