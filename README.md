# Bird Classification, Layer by Layer
## Overview
This project seeks to train a convolutional neural network to identify 18 bird species. 
We will compare the CNN's inner layers to MRI scans from humans attempting the same task.

## Our Model
Our base model is a Resnet-50 model pretrained on ImageNet. 
We then train the model on 200 bird species from the [Caltech 2011 dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/). 
Finally, we fine-tune using 75% of the photos from our human study and evaluate our model on the rest.

## Results so Far
Our model reaches over 90% accuracy on this classification task, in line with the best experts in our study.
The model seems to confuse the same birds that human experts do.
We are in the process of extracting activation layers from our model for certain 'cue' images to compare them with MRI scan data.

## Running the Project
- The file `main.py` trains Resnet-50 on 200 image classes from the [Caltech 2011 dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/).
- The file `mribirds.py` trains Resnet-50 on our 18 image classes.
- The file `test_model.py` calculates the accuracy of a model and generates a confusion matrix for the 18 species. This confusion matrix will be compared with the confusion matrices for both human experts and novices.
- The file `extractfeatures.py` extracts the activations for a specified internal layer and saves them to a hdf5 file.

## Running in Slurm
If using Slurm, run `./run_slurm.sh` to run our model.

Alternatively, start an interactive Slurm session and run `python main.py` to get started.


