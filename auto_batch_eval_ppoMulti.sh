#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=pick_train_analysis
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
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
LOG_PATH="/home/bahaduri/VIPER/outputs/ppo_LLM_dual_dcy_tmp_logs/"
VLM=False
TASK=1
# MODELS=("intrinsic")   #"llm"    #"LLM_dual_dcy_tmp_2" 
MODELS=("LLM_dual_dcy_tmp_1" "LLM_dual_dcy_tmp_2" "LLM_dual_dcy_tmp_3" "LLM_dual_dcy_tmp_4")
PORT=12321

for MODEL in "${MODELS[@]}"; do
    output="/home/bahaduri/VIPER/outputs/"
    output_dir="$output$MODEL"
    epoch_dirs=($(find "$output_dir" -mindepth 1 -maxdepth 1 -type d | sort -V))
    ext=".txt"
    RESULTS_FILE="$LOG_PATH$MODEL$ext"      #log_path/model.txt
    count=0
    for dir in "${epoch_dirs[@]}"; do
        if (( count % 5 == 0 )); then
            echo "Evaluating $dir"
            MODEL_PATH="$dir"
            TASK_NAME="$count"
            # TASK_NAME is just epoch number here
            bash eval_auto.sh  "$TASK" "$VLM" "$MODEL_PATH" "$RESULTS_FILE" "$TASK_NAME" "$PORT" "$LOG_PATH"
        fi
        ((count++))
    done
done