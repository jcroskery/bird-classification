# This file lists all images in a folder in alphabetical order, using the Torchvision data loader
# If seeking to regenerate images.txt, use list_birds.py instead

import os
from torchvision import datasets

# Folder path
folder_path = './mribirdsdata/images'

# Use ImageFolder to load the dataset
dataset = datasets.ImageFolder(root=folder_path)

# Traverse through the dataset and print paths to each image
for img_path, _ in dataset.imgs:
    # Split the path into components
    path_components = img_path.split(os.sep)
    # Remove the first two components (folders)
    relative_path = os.path.join(*path_components[3:])
    print(relative_path)
