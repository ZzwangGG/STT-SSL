#!/bin/bash

# 检查参数
if [ "$#" -ne 2 ]; then
    echo "用法: ./runme <GPU_ID> <DATASET>"
    echo "示例: ./runme 0 PEMS08-12"
    exit 1
fi

# 参数
GPU_ID=$1
DATASET=$2
CONFIG_FILE="configs/${DATASET}.yaml"

# 设置 GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 启动训练
echo "使用 GPU: $GPU_ID, 配置文件: $CONFIG_FILE"
python main.py --config_filename $CONFIG_FILE
