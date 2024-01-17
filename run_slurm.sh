#!/bin/sh
#SBATCH --time=01:00:00
#SBATCH --account=def-agilboa
#SBATCH --mem 16G
#SBATCH --cpus-per-task=16

cd $SLURM_TMPDIR
git clone https://github.com/jcroskery/caltech-birds-advanced-classification
cd caltech-birds-advanced-classification
python ./main.py
