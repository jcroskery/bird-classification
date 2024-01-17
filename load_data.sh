#!/bin/sh

wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
mkdir data
tar zxvf CUB_200_2011.tgz -C ./data/
