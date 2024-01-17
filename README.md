# Bird Classifier - Version 1
## Running in Slurm
If using Slurm, run `./run_slurm.sh` to run our model.
This clones our repository into `$SLURM_TMPDIR` and executes main.py.

## Docker Installation
First, build the docker image from the Dockerfile inside this repository using the command `docker build --tag 'birds'`.

Then, run the docker image using `docker run -d 'birds'`.

We can check the logs using `docker logs -f <CONTAINER>`.
