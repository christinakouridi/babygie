# base image (CUDA/cudNN)
FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV CUDA cu102
ENV TORCH 1.6.0
ENV BABYAI_STORAGE /models

RUN mkdir /models

# install python
RUN apt-get update -y
RUN apt-get install -y qt5-default qttools5-dev-tools git python3 python3-pip
RUN pip3 install --upgrade pip

# install poetry and setup env
COPY requirements.txt .
RUN pip install torch torchvision \
    pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html \
    pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html \
    pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html \
    pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html \
    pip install torch-geometric \
    pip install --ignore-installed -r requirements.txt

# install local gym-minigrid edits
COPY gym-minigrid /gym-minigrid/
RUN pip install /gym-minigrid

# code
WORKDIR /
COPY babyai /babygie/babyai/
COPY scripts /babygie/scripts/

# train
WORKDIR /babygie
