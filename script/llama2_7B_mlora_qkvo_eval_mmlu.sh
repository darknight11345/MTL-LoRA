#!/usr/bin/env bash
#SBATCH --partition=dev_gpu_h100        # Dev-Queue (H100)
#SBATCH --gres=gpu:2                    # 1 × H100
#SBATCH --cpus-per-task=8               # 8 CPU-Kerne
#SBATCH --mem=64G                       # 64 GB RAM
#SBATCH -t 00:30:00                     # 20 Min Testlauf
#SBATCH -J datascience_gpu_dev              # Job-Name


OUTPUT_PATH="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/output"  # Output directory is not used in this script
SCRIPT_PATH="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/mlora_evaluate_mmlu.py"
CHECKPOINT_PATH_MMLU="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/output/MMLU/checkpoint/final_checkpoint.pt"
OUTPUT_PATH_MMLU="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/output/eval_output/mmlu"


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
 

CUDA_VISIBLE_DEVICES=1 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset econometrics \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 11 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH_MMLU|tee -a $OUTPUT_PATH_MMLU/econometrics.txt \
    &

CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset high_school_macroeconomics \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 11 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH_MMLU|tee -a $OUTPUT_PATH_MMLU/high_school_macroeconomics.txt \
    &



CUDA_VISIBLE_DEVICES=1 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset high_school_microeconomics \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 11 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH_MMLU|tee -a $OUTPUT_PATH_MMLU/high_school_microeconomics.txt \
    &

CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset business_ethics \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 11 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH_MMLU|tee -a $OUTPUT_PATH_MMLU/business_ethics.txt \
    &

CUDA_VISIBLE_DEVICES=1 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset management \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 11 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH_MMLU|tee -a $OUTPUT_PATH_MMLU/management.txt \
    &



CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset marketing \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 11 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH_MMLU|tee -a $OUTPUT_PATH_MMLU/marketing.txt \
    &

CUDA_VISIBLE_DEVICES=1 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset abstract_algebra \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 11 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH_MMLU|tee -a $OUTPUT_PATH_MMLU/abstract_algebra.txt \
    &

CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset college_mathematics \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 11 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH_MMLU|tee -a $OUTPUT_PATH_MMLU/college_mathematics.txt \
    &


CUDA_VISIBLE_DEVICES=1 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset elementary_mathematics \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 11 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH_MMLU|tee -a $OUTPUT_PATH_MMLU/elementary_mathematics.txt \
    &

CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset high_school_mathematics \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 11 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH_MMLU|tee -a $OUTPUT_PATH_MMLU/high_school_mathematics.txt \
    &

CUDA_VISIBLE_DEVICES=1 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset high_school_statistics \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 11 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH_MMLU|tee -a $OUTPUT_PATH_MMLU/high_school_statistics.txt \
    &





wait

echo "Done"
