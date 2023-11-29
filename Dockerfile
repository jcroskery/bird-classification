FROM cnstark/pytorch:2.0.1-py3.10.11-cuda11.8.0-ubuntu22.04

WORKDIR /usr/src/app

RUN pip install torch
RUN pip install torchvision
RUN pip install numpy
RUN pip install scikit-learn

COPY ./main.py .

RUN wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
RUN mkdir data
RUN tar zxvf CUB_200_2011.tgz -C ./data/
CMD python ./main.py
