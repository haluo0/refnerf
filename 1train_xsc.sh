#!/bin/bash
#SBATCH --job-name=tex
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load anaconda/3.6
source activate /opt/ohpc/pub/apps/pytorch_1.6.0
module load cuda/10.2
module load gnu/5.4.0
module load mvapich2

#pip install --user torchinfo
#pip install --user opencv-python==4.2.0.34
#pip install --user timm
#pip install --user configargparse
srun python3 ./train.py
