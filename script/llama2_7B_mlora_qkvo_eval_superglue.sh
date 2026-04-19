#!/usr/bin/env bash
#SBATCH --partition=gpu_h100        # Dev-Queue (H100)
#SBATCH --gres=gpu:2                    # 1 × H100
#SBATCH --cpus-per-task=48               # 8 CPU-Kerne
#SBATCH --mem=128G                       # 64 GB RAM
#SBATCH -t 05:00:00                     # 20 Min Testlauf
#SBATCH -J datascience_gpu_dev              # Job-Name


OUTPUT_PATH="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/output"  # Output directory is not used in this script
SCRIPT_PATH="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/mlora_evaluate_superglue.py"
CHECKPOINT_PATH="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/output/SuperGLUE/checkpoint/checkpoint/final_checkpoint.pt"
OUTPUT_PATH="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/output/eval_output/superglue"
#/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/output/SuperGLUE/checkpoint/checkpoint/final_checkpoint.pt

#/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/output/SuperGLUE/checkpoint/checkpoint/final_checkpoint.pt
## 2) CUDA-Modul -------------------------------------------------------
module load devel/cuda/12.8              # laut `module avail cuda`

## 3) Kurz-Check (ASCII-nur, ohne Umlaut) ------------------------------
python - <<'PY'
#import importlib, vllm, torch
import importlib, torch
#print("vllm im Pfad:", bool(importlib.util.find_spec("vllm")))
print("CUDA available:", torch.cuda.is_available())
PY
 



CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset multirc \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 8 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/multirc.txt \
    &

CUDA_VISIBLE_DEVICES=1 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset wsc \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 8 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/wsc.txt \
    &

CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset boolq \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 8 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/boolq.txt \
    &

CUDA_VISIBLE_DEVICES=1 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset cb \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 8 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/cb.txt \
    &

CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset wic \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 8 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/wic.txt \
    &



wait

echo "Done"
