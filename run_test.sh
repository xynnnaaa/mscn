#!/bin/bash

# LRS=(0.01 0.005 0.002 0.001 0.0005 0.0002 0.0001 0.00005)
# LRS=(0.00005 0.0001 0.0002)

# new
# LRS=(0.002 0.001 0.0005)
# LRS=(0.0002 0.0001 0.00005)

LRS=(0.0002 0.0001 0.00005 0.002 0.001 0.0005)

CONFIG_PATH="./data/imdb/config-join.json"

GPU_ID=2

echo "Starting parallel training on GPU ${GPU_ID}..."

for lr in "${LRS[@]}"
do
    LOG_FILE="./data/imdb/runfile/new-model/join-${lr}.log"
    
    echo "Running experiment with LR: ${lr}, logging to: ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python3 -u train.py \
        --config ${CONFIG_PATH} \
        --lr ${lr} \
        > ${LOG_FILE} 2>&1 &

    sleep 2
done

echo "All experiments have been submitted. Use 'top' or 'nvidia-smi' to monitor."