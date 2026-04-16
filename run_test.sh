#!/bin/bash

# 设定的学习率列表
# 0.01: 极快，看是否能快速收敛
# 0.005, 0.002: 略高于默认，寻找更快下降点
# 0.001: MSCN 默认值
# 0.0005, 0.0001, 0.00005: 针对 Embedding 优化，更稳健，防止 Q-error 炸开

# LRS=(0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001 0.00005)
# LRS=(0.00005 0.0001 0.0002)

# new
# LRS=(0.002 0.001 0.0005)
# LRS=(0.0002 0.0001 0.00005)

LRS=(0.0002 0.0001 0.00005 0.002 0.001 0.0005)

# 配置文件路径
CONFIG_PATH="./data/stats/config-default.json"

# 指定使用的 GPU ID
GPU_ID=0

echo "Starting parallel training on GPU ${GPU_ID}..."

for lr in "${LRS[@]}"
do
    # 生成日志文件名
    LOG_FILE="./data/stats/runfile/default-${lr}-t5-all.log"
    
    echo "Running experiment with LR: ${lr}, logging to: ${LOG_FILE}"
    
    # 核心执行命令
    # 使用 CUDA_VISIBLE_DEVICES 指定显卡
    # 使用 nohup 确保后台运行
    # 使用 python3 -u 确保日志实时刷新不缓冲
    CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python3 -u train.py \
        --config ${CONFIG_PATH} \
        --lr ${lr} \
        > ${LOG_FILE} 2>&1 &
    
    # 稍微延迟一下启动，防止多个进程同时读写文件系统造成竞争
    sleep 2
done

echo "All experiments have been submitted. Use 'top' or 'nvidia-smi' to monitor."