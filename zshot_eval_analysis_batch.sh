#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=pick_train_analysis
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=42:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
nvidia-smi -L
source /usr/local/miniconda/etc/profile.d/conda.sh
cd /home/bahaduri/VIPER

conda activate viper2
# TASKS=(1 2 3 4 5 6)
TASKS=(1 3)
# TASK_NAMES=("pick" "examine" "clean" "heat" "cool" "pick2")
TRIALS=(1 2 3 4 5)
TASK_NAMES=("pick" "clean")
NUM_TASKS=${#TASKS[@]}
NUM_TRIALS=${#TRIALS[@]}
USE_LORA=(False)
RESULTS_FILE="/home/bahaduri/VIPER/outputs/full_pick_lora.txt"
MODELS=("mistral7b" "llama1b")

for MODEL in "${MODELS[@]}"; do
    for ((i=0; i<NUM_TRIALS; i++)); do
        TASK=${TASKS[0]}
        TASK_NAME=${TASK_NAMES[0]}
        TRIAL=${TRIALS[$i]}
        for LORA in "${USE_LORA[@]}"; do

            LOG_PATH="/home/bahaduri/VIPER/outputs/full_eval_${TASK}_LORA_${LORA}_model_${MODEL}"
            JSON_PATH="/home/bahaduri/VIPER/outputs/full_eval_${TASK}_LORA_${LORA}_model_${MODEL}_tr_${TRIAL}.json"
            if [[ "$MODEL" == "mistral7b" ]]; then
                MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.3"
            fi
            if [[ "$MODEL" == "llama1b" ]]; then
                MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
            fi 
            #create log_path if not existing
            if [ ! -d "$LOG_PATH" ]; then
                mkdir -p "$LOG_PATH"
                echo "Created directory: $LOG_PATH"
            fi
            echo "Running Task $TASK ($TASK_NAME) with LORA: $LORA MODEL: $MODEL"
            bash eval_zshot_analysis.sh  "$TASK" "$LORA" "$MODEL_PATH" "$LOG_PATH" "$RESULTS_FILE" "$TASK_NAME" "$JSON_PATH"
        done
    done
done