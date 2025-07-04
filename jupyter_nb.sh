#!/bin/bash

#SBATCH --job-name=visu

#SBATCH --partition=funky

#SBATCH --nodes=1

#SBATCH --time=02:00:00

#SBATCH --gpus-per-node=1

#SBATCH --output=visualisation.out

#SBATCH --error=visualisation.err
export http_proxy=http://"192.168.0.100":"3128"
export https_proxy=http://"192.168.0.100":"3128"
echo debut
source /usr/local/miniconda/etc/profile.d/conda.sh


conda activate viper2
which jupyter-notebook 
##conda env list
jupyter notebook --no-browser --ip=0.0.0.0 --port=12356