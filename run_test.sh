
LRS=(0.00005 0.0001 0.0002 0.0005 0.001 0.002)

CONFIG_PATH="./data/imdb/runfile_ablation/change_pca/pca-3-mean/config-single-pca.json"
GPUS=(1 1 2)

echo "Starting parallel training on GPUs ${GPUS[*]}..."

for i in "${!LRS[@]}"
do
    lr=${LRS[$i]}
    GPU_ID=${GPUS[$((i % ${#GPUS[@]}))]}
    LOG_FILE="./data/imdb/runfile_ablation/change_pca/pca-3-mean/${lr}.log"
    
    echo "Running experiment with LR: ${lr} on GPU ${GPU_ID}, logging to: ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python3 -u train.py \
        --config ${CONFIG_PATH} \
        --lr ${lr} \
        > ${LOG_FILE} 2>&1 &

    sleep 3
done

echo "All experiments have been submitted. Use 'top' or 'nvidia-smi' to monitor."



# #!/bin/bash

# LRS=(0.00005 0.0001 0.0002 0.0005 0.001 0.002)
# CONFIG_PATH="./data/stats/config-single-pca.json"
# GPU_ID=1   # 指定要使用的 GPU 编号

# echo "Starting serial training on GPU ${GPU_ID}..."

# for lr in "${LRS[@]}"
# do
#     LOG_FILE="./data/stats/runfile/no-join-1/${lr}-128.log"
#     echo "Running experiment with LR: ${lr} on GPU ${GPU_ID}, logging to: ${LOG_FILE}"
    
#     CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -u train.py \
#         --config ${CONFIG_PATH} \
#         --lr ${lr} \
#         > ${LOG_FILE} 2>&1

#     echo "Finished experiment with LR: ${lr}"
# done

# echo "All experiments completed."