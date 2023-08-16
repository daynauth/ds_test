FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}

##############################################################################
# Installation/Basic Utilities
##############################################################################
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc \
        curl wget vim tmux emacs less unzip \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo \
        llvm-9-dev

##############################################################################
# Installation Latest Git
##############################################################################
RUN add-apt-repository ppa:git-core/ppa -y && \
        apt-get update && \
        apt-get install -y git && \
        git --version

##############################################################################
# Python
##############################################################################
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3
RUN apt-get install -y python3 python3-dev && \
        rm -f /usr/bin/python && \
        ln -s /usr/bin/python3 /usr/bin/python
RUN apt-get install -y python3-pip && \
        pip3 install --upgrade pip



##############################################################################
# PyTorch
##############################################################################
ENV PYTORCH_VERSION=1.13.1+cu117
ENV TORCHVISION_VERSION=0.14.1+cu117
ENV TORCH_AUDIO_VERSION=0.13.1

RUN pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCH_AUDIO_VERSION} --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install transformers


##############################################################################
# PyYAML build issue
# https://stackoverflow.com/a/53926898
##############################################################################
# RUN rm -rf /usr/lib/python3/dist-packages/yaml && \
#         rm -rf /usr/lib/python3/dist-packages/PyYAML-*

##############################################################################
## Add deepspeed user
###############################################################################
# Add a deepspeed user with user id 8877
RUN useradd --create-home --uid 1000 --shell /bin/bash deepspeed
RUN usermod -aG sudo deepspeed
RUN echo "deepspeed ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
# # Change to non-root privilege
USER deepspeed




##############################################################################
# DeepSpeed
##############################################################################
ENV DS_BUILD_TRANSFORMER_INFERENCE=1 
RUN pip install deepspeed --global-option="build_ext"

WORKDIR /home/deepspeed

RUN pip install pandas
ENV PATH="${PATH}:/home/deepspeed/.local/bin"

