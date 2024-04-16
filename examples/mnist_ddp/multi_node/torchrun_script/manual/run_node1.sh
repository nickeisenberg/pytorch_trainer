torchrun \
    --nproc_per_node=gpu \
    --nnodes=2 \
    --node_rank=1 \
    --rdzv_id=456 \
    --rdzc_backend=c10d \
    --rdzv_endpoint=<host>:<port> \
    ../../main.py
