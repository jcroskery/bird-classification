#!/bin/sh

cd caltech-birds-advanced-classification
module load StdEnv/2020
module load python/3.10
virtualenv --no-download .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# After the environment is loaded, switch to the compute node
salloc --time=1:0:0 --nodes=1 --cpus-per-task=2 --mem=16G --gres=gpu:1 --account=def-agilboa
