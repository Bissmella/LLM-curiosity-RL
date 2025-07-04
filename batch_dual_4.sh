#!/bin/bash
#SBATCH --partition=hard
#SBATCH --exclude=led,lizzy,thin,zeppelin
#SBATCH --job-name=pick_train_analysis
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=42:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
nvidia-smi -L
source /usr/local/miniconda/etc/profile.d/conda.sh
cd /home/bahaduri/VIPER


#"aerosmith,top,zz"
conda activate viper2
SEED=4
WANDBRUN="LLM_dual_4"
OUTPUT_DIR="/home/bahaduri/VIPER/outputs/LLM_dual_4"  #$2
PORT=12360
bash auto_run_dual.sh "$SEED" "$WANDBRUN" "$OUTPUT_DIR" "$PORT"