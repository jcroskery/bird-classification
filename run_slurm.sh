#!/bin/sh
#SBATCH --time=03:00:00
#SBATCH --account=def-agilboa
#SBATCH --mem 16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1 

cd caltech-birds-advanced-classification
module load python/3.11.5
python ./main.py
