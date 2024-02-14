# Bird Classifier - Version 1
## Running the Project
The file `main.py` trains Resnet-50 on 200 image classes from the Caltech dataset.
The file `mribirds.py` trains Resnet-50 on our 18 image classes from the MRI study.
The file `test_model.py` calculates the accuracy of a model and generates a confusion matrix.

## Running in Slurm
If using Slurm, run `./run_slurm.sh` to run our model.

## Docker Installation
First, build the docker image from the Dockerfile inside this repository using the command `docker build --tag 'birds'`.

Then, run the docker image using `docker run -d 'birds'`.

We can check the logs using `docker logs -f <CONTAINER>`.
