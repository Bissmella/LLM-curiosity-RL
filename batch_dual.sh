#!/bin/bash
#SBATCH --partition=hard
#SBATCH --nodelist=aerosmith
#SBATCH --job-name=pick_train_analysis
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=36:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
nvidia-smi -L
source /usr/local/miniconda/etc/profile.d/conda.sh
cd /home/bahaduri/VIPER


#"aerosmith,top,zz"
conda activate viper2
SEED=1
WANDBRUN="LLM_dual_1"
OUTPUT_DIR="/home/bahaduri/VIPER/outputs/LLM_dual_1"  #$2  
bash auto_run_dual.sh "$SEED" "$WANDBRUN" "$OUTPUT_DIR"