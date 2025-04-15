#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=pick_train_analysis
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=14:00:00
#SBATCH –mail-type=ALL
#SBATCH –mail-user=bahaduri@isir.upmc.fr
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
nvidia-smi -L

sleep 100
cd /home/bahaduri/VIPER
conda activate viper2
sh eval.sh