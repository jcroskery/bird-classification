#!/bin/sh
#SBATCH --time=01:00:00
#SBATCH --account=def-agilboa
#SBATCH --mem 32G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1 

cd caltech-birds-advanced-classification
module load python/3.10
virtualenv --no-download .venv
source .venv/bin/activate
