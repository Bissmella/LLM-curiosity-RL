#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=pick_train_analysis
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=28:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
nvidia-smi -L
source /usr/local/miniconda/etc/profile.d/conda.sh
cd /home/bahaduri/VIPER

conda activate viper2
TASKS=(1 2 3 4 5 6)
TASK_NAMES=("pick" "look" "clean" "heat" "cool" "pick2")
NUM_TASKS=${#TASKS[@]}
USE_VLM=(True False)
RESULTS_FILE="/home/bahaduri/VIPER/outputs/full_eval_results.txt"
for ((i=0; i<NUM_TASKS; i++)); do
    TASK=${TASKS[$i]}
    TASK_NAME=${TASK_NAMES[$i]}
    MODEL_PATH="/home/bahaduri/VIPER/weights/Mistral/${TASK_NAME}"
    for VLM in "${USE_VLM[@]}"; do

        LOG_PATH="/home/bahaduri/VIPER/outputs/full_eval_${TASK}_VLM_${VLM}"
        JSON_PATH="/home/bahaduri/VIPER/outputs/full_eval_${TASK}_VLM_${VLM}.json"
        #create log_path if not existing
        if [ ! -d "$LOG_PATH" ]; then
            mkdir -p "$LOG_PATH"
            echo "Created directory: $LOG_PATH"
        fi
        echo "Running Task $TASK ($TASK_NAME) with VLM: $VLM"
        bash eval_auto.sh  "$TASK" "$VLM" "$MODEL_PATH" "$LOG_PATH" "$RESULTS_FILE" "$TASK_NAME" "$JSON_PATH"
    done
done