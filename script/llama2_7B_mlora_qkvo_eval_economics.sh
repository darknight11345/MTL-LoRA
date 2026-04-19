#!/usr/bin/env bash
#SBATCH --partition=gpu_h100        # Dev-Queue (H100)
#SBATCH --gres=gpu:2                    # 1 × H100
#SBATCH --cpus-per-task=48               # 8 CPU-Kerne
#SBATCH --mem=380G                       # 64 GB RAM
#SBATCH -t 60:00:00                     # 20 Min Testlauf
#SBATCH -J datascience_gpu_dev              # Job-Name


OUTPUT_PATH="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/output"  # Output directory is not used in this script
SCRIPT_PATH="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/mlora_evaluate_economics.py"
CHECKPOINT_PATH="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/output/economics/checkpoint/final_checkpoint.pt"
OUTPUT_PATH="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/output/eval_output/economics"



#module load devel/miniforge/24.11.0-python-3.12
#export PATH="$WS_MODEL/conda/pixtral/bin:$PATH"
export HF_TOKEN=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
echo $HF_TOKEN


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
    --dataset car_combined \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 7 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/car_combined.txt \
    &

CUDA_VISIBLE_DEVICES=1 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset car_summary \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 7 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/car_summary.txt \
    &

CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset sentiment \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 7 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/sentiment.txt \
    &

CUDA_VISIBLE_DEVICES=1 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset car_combined_us \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 7 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/car_combined_us.txt \
    &

CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset car_combined_eu \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 7 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/car_combined_eu.txt \
    &

CUDA_VISIBLE_DEVICES=1 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset sentiment_us \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 7 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/sentiment_us.txt \
    &

CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset sentiment_eu \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 7 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/sentiment_eu.txt \
    &


wait

echo "Done"
