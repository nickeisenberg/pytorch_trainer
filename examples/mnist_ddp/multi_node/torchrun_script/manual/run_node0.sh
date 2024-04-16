torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --rdzv_id=456 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=B301486:8000 \
    /home/nick/GitRepos/torchface/examples/mnist_ddp/multi_node/main.py
