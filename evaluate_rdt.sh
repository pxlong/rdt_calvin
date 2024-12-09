#!/bin/bash

# export MESA_GL_VERSION_OVERRIDE=3.3

python3 evaluate_calvin.py \
    --eval_dir exps/calvin/task_d_d \
    --configs_path configs/eval_configs.json \
    --dataset_dir /mnt/petrelfs/longpinxin/data/calvin \
    ${@:1}