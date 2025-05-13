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

bash eval.sh