#!/usr/bin/env bash
#SBATCH --partition=gpu_h100        # Dev-Queue (H100)
#SBATCH --gres=gpu:2                    # 1 × H100
#SBATCH --cpus-per-task=48               # 8 CPU-Kerne
#SBATCH --mem=128G                       # 64 GB RAM
#SBATCH -t 05:00:00                     # 20 Min Testlauf
#SBATCH -J datascience_gpu_dev              # Job-Name


OUTPUT_PATH="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/output"  # Output directory is not used in this script
SCRIPT_PATH="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/mlora_evaluate_bigbench.py"
CHECKPOINT_PATH="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/output/bigbench/checkpoint/checkpoint/final_checkpoint.pt"
OUTPUT_PATH="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/output/eval_output/bigbench"
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
    --dataset all_sentences_en \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 9 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/all_sentences_en.txt \
    &

CUDA_VISIBLE_DEVICES=1 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset all_text_en \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 9 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/all_text_en.txt \
    &

CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset causal_judgment \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 9 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/causal_judgment.txt \
    &

CUDA_VISIBLE_DEVICES=1 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset formal_fallacies \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 9 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/formal_fallacies.txt \
    &

CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset logical_fallacy \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 9 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/logical_fallacy.txt \
    &

CUDA_VISIBLE_DEVICES=1 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset figure_of_speech \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 9 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/figure_of_speech.txt \
    &

CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset stratergyqa \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 9 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/stratergyqa.txt \
    &
CUDA_VISIBLE_DEVICES=1 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset navigate \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 9 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/navigate.txt \
    &

CUDA_VISIBLE_DEVICES=0 python $SCRIPT_PATH \
    --model LLaMA-7B \
    --adapter mlora \
    --dataset information_essentiality \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --batch_size 1 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lambda_num 9 \
    --num_B 3 \
    --temperature 0.1 \
    --lora_weights $CHECKPOINT_PATH|tee -a $OUTPUT_PATH/information_essentiality.txt \
    &


wait

echo "Done"
