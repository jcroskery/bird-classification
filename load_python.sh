#!/bin/sh
#SBATCH --time=01:00:00
#SBATCH --account=crosker4
#SBATCH --mem 16G
#SBATCH --cpus-per-task=8

module load python/3.11.5
pip install -r requirements.txt --no-index
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
mkdir data
tar zxvf CUB_200_2011.tgz -C ./data/
python ./main.py

