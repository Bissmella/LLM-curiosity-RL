#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=pick_ppo_wintrinsic
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=46:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
nvidia-smi -L
source /usr/local/miniconda/etc/profile.d/conda.sh
cd /home/bahaduri/VIPER

conda activate viper2
# TASKS=(1 2 3 4 5 6)
TASKS=(1) # 3)
# TASK_NAMES=("pick" "examine" "clean" "heat" "cool" "pick2")
TASK_NAMES=("pick" "clean")
NUM_TASKS=${#TASKS[@]}
RESULTS_FILE="/home/bahaduri/VIPER/outputs/full_eval_results_pick_examine.txt"
for ((i=0; i<NUM_TASKS; i++)); do
    TASK=${TASKS[$i]}
    TASK_NAME=${TASK_NAMES[$i]}
    #MODEL_PATH="/home/bahaduri/VIPER/weights/Mistral/${TASK_NAME}"

        
    #LOG_PATH="/home/bahaduri/VIPER/outputs/full_eval_${TASK}_VLM_${VLM}"
    JSON_PATH="/home/bahaduri/VIPER/outputs/full_eval_${TASK}_VLM_${VLM}.json"
    OUTPUT_DIR="/home/bahaduri/VIPER/outputs/ppo_llm_wIntrinsic_40gam_dual_${TASK_NAME}"
    #create log_path if not existing
    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"
        echo "Created directory: $OUTPUT_DIR"
    fi
    echo "Running Task $TASK ($TASK_NAME) with VLM: $VLM"
    bash auto_run.sh  "$TASK" "$OUTPUT_DIR"
done