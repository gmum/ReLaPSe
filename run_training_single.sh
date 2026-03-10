#!/bin/bash

UNET="ESD"
# UNET="EraseDiff"
# UNET="FMN"
# UNET="SPM"
# UNET="Salun"
# UNET="Scissorhands"

OBJECT="Nudity"
# OBJECT="Church"
# OBJECT="Garbage_Truck"
# OBJECT="Tench"
# OBJECT="Parachute"


case "${OBJECT}" in
  Church) OBJDIR="church" ;;
  Garbage_Truck) OBJDIR="garbage_truck" ;;
  Parachute) OBJDIR="parachute" ;;
  Tench) OBJDIR="tench" ;;
  VanGogh) OBJDIR="vangogh" ;;
  Nudity) OBJDIR="nudity" ;;
  *) echo "Unknown OBJECT=${OBJECT}" >&2; exit 1 ;;
esac

REWARD_WEIGHTS=(1.0 0.0001)
TEMPERATURE=1.0
TOP_P=0.95
TOP_K=80
NUM_GENERATIONS=8
BETA=0.01
EPSILON=0.24
LR=2e-5

MODEL_QUANT="7"
MODEL="Qwen/Qwen2.5-${MODEL_QUANT}B-Instruct"
BASE_MODEL="CompVis/stable-diffusion-v1-4"
UNET_PATH="files/models/${UNET}-${OBJECT}-Diffusers-UNet-noxattn.pt" # UNET_PATH="files/models/${UNET}-${OBJECT}-Diffusers-UNet.pt"
BASE_OUTPUT="runs/${UNET}_${OBJECT}/single"
mkdir -p "$BASE_OUTPUT"

DATASET_DIR="files/datasets/${OBJDIR}/single"
for IDX in {0..4}; do
  DATA_FILE="${DATASET_DIR}/example_${IDX}.jsonl"
  RUN_DIR="${BASE_OUTPUT}/run_${IDX}"
  mkdir -p "$RUN_DIR"
  
  swift rlhf \
    --rlhf_type grpo \
    --model "${MODEL}" \
    --train_type lora \
    --torch_dtype bfloat16 \
    --reward_funcs "denoising_${OBJDIR}" format \
    --reward_weights "${REWARD_WEIGHTS[@]}" \
    --base_model_name "${BASE_MODEL}" \
    --unlearned_unet_path "${UNET_PATH}" \
    --dataset "${DATA_FILE}" \
    --remove_unused_columns false \
    --num_generations "${NUM_GENERATIONS}" \
    --max_completion_length 512 \
    --max_length 2048 \
    --num_train_epochs 1000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate "${LR}" \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 5 \
    --output_dir "${RUN_DIR}" \
    --report_to wandb \
    --log_completions true \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --epsilon "${EPSILON}" \
    --beta "${BETA}" \
    --seed 0 
done