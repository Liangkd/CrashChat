#!/bin/bash
# Usage examples:
# Use only GPU1 (logical 0): CUDA_VISIBLE_DEVICES=0 bash /CrashChat/scripts/train/Homogeneous_multitask_models_perception.sh 1
# Use GPU1 + GPU2:      CUDA_VISIBLE_DEVICES=1,2 bash /CrashChat/scripts/train/Homogeneous_multitask_models_perception.sh 2

#########################
# ===== Node & Process Settings =====
#########################
WORLD_SIZE=1                       # Just one machine.
NPROC_PER_NODE=${1:-1}             # Uses several cards, controlled by the first parameter (default: 1).

MASTER_ADDR="127.0.0.1"
MASTER_PORT=16704
RANK=0

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

#########################
# ===== Batch & Cumulative Steps =====
#########################
GLOBAL_BATCH_SIZE=24               
LOCAL_BATCH_SIZE=24                
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]
echo "GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"

######################### 
# ===== Log and path settings =====
#########################
export WANDB_PROJECT=crashchat_7B_homogeneous_perception
RUN_NAME=homogeneous_multitask_models_perception
MODEL_PATH=ckpt/videollama3_baseline
DATA_DIR=data
TRAIN_PATH=${DATA_DIR}/crashchat_dada_video_total_perception_centric_train.json
VAL_PATH=${DATA_DIR}/crashchat_dada_video_total_perception_centric_val.json
OUTP_DIR=ckpt

#########################
# ===== Start training =====
#########################

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    videollama3/train.py \
    --deepspeed scripts/zero1.json \
    --model_type videollama3_qwen2 \
    --model_path ${MODEL_PATH} \
    --vision_encoder DAMO-NLP-SG/SigLIP-NaViT \
    --mm_projector_type mlp2x_gelu \
    --data_path ${TRAIN_PATH} \
    --eval_data_path ${VAL_PATH} \
    --data_folder ${DATA_DIR} \
    --image_merge_size 2 \
    --video_merge_size 2 \
    --fps 10 \
    --max_frames 96 \
    --model_max_length 16384 \
    --mm_max_length 10240 \
    --use_token_compression True \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/${RUN_NAME} \
    --num_train_epochs 10 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 24 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 5 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    --llm_lr 1e-5 \
    --mm_projector_lr 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --report_to wandb \
    --run_name $RUN_NAME \
    --dataset_cache_dir .cache \
    --freeze_backbone True \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_bias "none" \
    --bits 16 \
    --early_stopping_patience 3 \
    --early_stopping_threshold 0.001     