#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

CONFIG=$1
WORK_DIR=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    /mnt/guodongqian/CL-Detection2023-MMPose/step3_train_and_evaluation.py \
    $CONFIG \
    --work-dir=$WORK_DIR \
    --launcher pytorch ${@:4}
