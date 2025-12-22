#!/bin/bash
#######################################
#
# Usage examples:
#   Single GPU：
#     CUDA_VISIBLE_DEVICES=1 bash scripts/eval/eval_video_crash_localization.sh.sh
#
#   Two GPUs：
#     CUDA_VISIBLE_DEVICES=0,1 bash scripts/eval/eval_video_crash_localization.sh.sh 2
#######################################

set -euo pipefail

# 1) First parameter: Number of GPUs to use (default 1)
NUM_GPUS=${1:-1}

#######################################
# 1. Model path
#######################################

MODEL_PATH="/ckpt/independent_monotask_models_crash_localization"

#######################################
# 2. Distributed Training (Single Machine, Multiple GPUs)
#######################################
WORLD_SIZE=1                    
NPROC_PER_NODE=${NUM_GPUS}      

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-26682}   
RANK=${RANK:-0}

echo "NUM_GPUS        : $NUM_GPUS"
echo "MODEL_PATH      : $MODEL_PATH"
echo "WORLD_SIZE      : $WORLD_SIZE"
echo "NPROC_PER_NODE  : $NPROC_PER_NODE"
echo "MASTER_ADDR     : $MASTER_ADDR"
echo "MASTER_PORT     : $MASTER_PORT"
echo "RANK            : $RANK"

#######################################
# 3. Data and output paths
####################################### 

DATA_ROOT="/data"

JSON_PATH="${DATA_ROOT}/crashchat_dada_video_total_crash_localization_test.json"

MODEL_NAME=$(basename "${MODEL_PATH}")
SAVE_DIR="/CrashChat/outputs/crash_localization_evaluation_results/${MODEL_NAME}"
mkdir -p "${SAVE_DIR}"

OUT_JSON="${SAVE_DIR}/crashchat_dada_video_total_crash_localization_test_predict.json"

echo "DATA_ROOT       : $DATA_ROOT"
echo "JSON_PATH       : $JSON_PATH"
echo "SAVE_DIR        : $SAVE_DIR"
echo "OUT_JSON        : $OUT_JSON"

#######################################
# 4. Calling evaluation/evaluate.py
#######################################

torchrun --nnodes ${WORLD_SIZE} \
         --nproc_per_node ${NPROC_PER_NODE} \
         --master_addr ${MASTER_ADDR} \
         --master_port ${MASTER_PORT} \
         --node_rank ${RANK} \
         evaluation/evaluate.py \
         --model-path "${MODEL_PATH}" \
         --benchmark "crash_tl" \
         --data-root "${DATA_ROOT}" \
         --save-path "${OUT_JSON}" \
         --num-workers 4 \
         --fps 10 \
         --max-frames 96 \
         --max-visual-tokens 16384
