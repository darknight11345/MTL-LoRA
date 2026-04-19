#!/usr/bin/env bash
#SBATCH --partition=gpu_h100        # Dev-Queue (H100)
#SBATCH --gres=gpu:1                    # 1 × H100
#SBATCH --cpus-per-task=24               # 8 CPU-Kerne 8 for dev_gpu_h100 and 24 for gpu_h100
#SBATCH --mem=64GB                       # 64 GB RAM for dev_gpu_h100 and 0 for gpu_h100
#SBATCH -t 31:00:00                     # 20 Min Testlauf 
#SBATCH -J datascience_gpu_dev              # Job-Name
#SBATCH --output=/pfs/work9/workspace/scratch/ul_swv79-MTLLoRA/slurm/slurm-%j.out
#SBATCH --error=/pfs/work9/workspace/scratch/ul_swv79-MTLLoRA/slurm/slurm-%j.err

SCRIPT_PATH="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/mlora_finetune.py"
DATA_PATH="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/dataset/bigbench_code_train_dataset.json"    # .../commonsense_170k_taskid.json
CACHE_DIR="/pfs/data6/home/ul/ul_student/ul_swv79/.cache/huggingface"    # Cache directory is not used in this script
DEEPSPEED_CONFIG="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/config/ds2.json"    #config/ds2.json
OUTPUT_PATH="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/output/bigbench/checkpoint"  # Output directory is not used in this script
#CHECKPOINT_PATH_RTE="/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/output/bigbench/checkpoint/final_checkpoint.pt"

#mkdir -p $TMPDIR/output
#OUTPUT_PATH="$TMPDIR/output"


#cp -r $DATA_PATH $TMPDIR/dataset.json
#DATA_PATH="$TMPDIR/dataset.json"


export HF_TOKEN=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
echo $HF_TOKEN

#export HF_HOME=$TMPDIR/hf_cache
#export TRANSFORMERS_CACHE=$TMPDIR/hf_cache
#export HF_DATASETS_CACHE=$TMPDIR/hf_cache


## 2) CUDA-Modul -------------------------------------------------------
module load devel/cuda/12.8              # laut `module avail cuda`

## 3) Kurz-Check (ASCII-nur, ohne Umlaut) ------------------------------
python - <<'PY'
#import importlib, vllm, torch
import importlib, torch
#print("vllm im Pfad:", bool(importlib.util.find_spec("vllm")))
print("CUDA available:", torch.cuda.is_available())
PY
 


unset WANDB_RUN_ID
unset WANDB_RUN_NAME

deepspeed \
    --master_port=25000 \
    $SCRIPT_PATH \
    --base_model 'meta-llama/Llama-2-7b-hf' \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --batch_size 16  \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --cutoff_len 256 \
    --save_step 10000  \
    --adapter_name mlora \
    --lora_target_modules '["q_proj", "k_proj", "v_proj", "o_proj"]' \
    --lora_r $1 \
    --lora_alpha $2 \
    --use_gradient_checkpointing \
    --lambda_num 9 \
    --num_B 3 \
    --temperature 0.1 \
    --cache_dir $CACHE_DIR \
    --deepspeed $DEEPSPEED_CONFIG 


#cp -r $TMPDIR/output /pfs/work9/workspace/scratch/ul_swv79-MTLLoRA/output