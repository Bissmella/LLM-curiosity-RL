#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=pick_train_analysis
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=28:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
nvidia-smi -L
source /usr/local/miniconda/etc/profile.d/conda.sh
cd /home/bahaduri/VIPER

conda activate viper2
# TASKS=(1 2 3 4 5 6)
TASKS=(1 2 3 4 5 6)
# TASK_NAMES=("pick" "examine" "clean" "heat" "cool" "pick2")
JSON_PATH=""
LOG_PATH=""
VLM=True
TASK=1
MODELS=("intrinsic")   #"llm" 
for MODEL in "${MODELS[@]}"; do
    if [[ "$MODEL" == "llm" ]]; then
        output_dir="/home/bahaduri/VIPER/outputs/ppo_llm_wEntropy_pick"
        epoch_dirs=($(find "$output_dir" -mindepth 1 -maxdepth 1 -type d | sort -V))
        RESULTS_FILE="/home/bahaduri/VIPER/outputs/ppo_train/pick_results_llm.txt"
    fi
    if [[ "$MODEL" == "intrinsic" ]]; then
        output_dir="/home/bahaduri/VIPER/outputs/ppo_llm_wIntrinsic_40gam_dual_pick"
        epoch_dirs=($(find "$output_dir" -mindepth 1 -maxdepth 1 -type d | sort -V))
        RESULTS_FILE="/home/bahaduri/VIPER/outputs/ppo_train/pick_results_intrinsic_dual_40gam.txt"
    fi
    count=0
    for dir in "${epoch_dirs[@]}"; do
        if (( count % 5 == 0 )); then
            echo "Evaluating $dir"
            MODEL_PATH="$dir"
            TASK_NAME="$count"
            # TASK_NAME is just epoch number here
            bash eval_auto.sh  "$TASK" "$VLM" "$MODEL_PATH" "$LOG_PATH" "$RESULTS_FILE" "$TASK_NAME" "$JSON_PATH"
        fi
        ((count++))
    done
done