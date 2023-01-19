# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu18.04

##############################################################################
# Temporary Installation Directory
##############################################################################
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}

##############################################################################
# Installation/Basic Utilities
##############################################################################
# config TZ to avoid hanging for Configuring tzdata
ENV TZ="America/New_York"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc \
        curl wget vim tmux less unzip \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo \
        iproute2 libtool autoconf apt-utils

##############################################################################
# Installation Latest Git
##############################################################################
RUN apt-get install -y -q software-properties-common
RUN add-apt-repository ppa:git-core/ppa -y
RUN apt-get update && apt-get install -y git && \
    git --version

##############################################################################
## Add deepspeed user
###############################################################################
# Add a deepspeed user with user id 8877
#RUN useradd --create-home --uid 8877 deepspeed
RUN useradd --create-home --uid 1000 --shell /bin/bash deepspeed
RUN usermod -aG sudo deepspeed
RUN echo "deepspeed ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
# # Change to non-root privilege
USER deepspeed

##############################################################################
# SSH Setup
##############################################################################
ENV USER=deepspeed
ENV HOME /home/$USER
ENV SSHDIR $HOME/.ssh
RUN mkdir -p ${SSHDIR} \
&& touch ${SSHDIR}/sshd_config \
&& ssh-keygen -t rsa -f ${SSHDIR}/ssh_host_rsa_key -N '' \
&& cp ${SSHDIR}/ssh_host_rsa_key.pub ${SSHDIR}/authorized_keys \
&& cp ${SSHDIR}/ssh_host_rsa_key ${SSHDIR}/id_rsa \
&& echo "       IdentityFile ${SSHDIR}/id_rsa" >> ${SSHDIR}/config \
&& echo "       StrictHostKeyChecking no" >> ${SSHDIR}/config \
&& echo "       UserKnownHostsFile /dev/null" >> ${SSHDIR}/config \
&& echo "       Port 2022" >> ${SSHDIR}/config \
&& echo 'Port 2022' >> ${SSHDIR}/sshd_config \
&& echo 'UsePrivilegeSeparation no' >> ${SSHDIR}/sshd_config \
&& echo "HostKey ${SSHDIR}/ssh_host_rsa_key" >> ${SSHDIR}/sshd_config \ && echo "PidFile ${SSHDIR}/sshd.pid" >> ${SSHDIR}/sshd_config \
&& chmod -R 600 ${SSHDIR}/* \
&& chown -R ${USER}:${USER} ${SSHDIR}/

RUN eval `ssh-agent -s` && ssh-add ${SSHDIR}/id_rsa

USER root

##############################################################################
# OPENMPI
##############################################################################
ENV OPENMPI_BASEVERSION=4.0
ENV OPENMPI_VERSION=${OPENMPI_BASEVERSION}.5
RUN cd ${STAGE_DIR} && \
    wget -q -O - https://download.open-mpi.org/release/open-mpi/v${OPENMPI_BASEVERSION}/openmpi-${OPENMPI_VERSION}.tar.gz | tar xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --prefix=/usr/local/openmpi-${OPENMPI_VERSION} && \
    make -j"$(nproc)" install && \
    ln -s /usr/local/openmpi-${OPENMPI_VERSION} /usr/local/mpi && \
    # Sanity check:
    test -f /usr/local/mpi/bin/mpic++ && \
    cd ${STAGE_DIR} && \
    rm -r ${STAGE_DIR}/openmpi-${OPENMPI_VERSION}
ENV PATH=/usr/local/mpi/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}
# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
    chmod a+x /usr/local/mpi/bin/mpirun

##############################################################################
# Python
##############################################################################
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-dev python3-pip
RUN apt-get install -y python3.8 python3.8-dev python3.8-venv
RUN rm -f /usr/bin/python && ln -s /usr/bin/python3.8 /usr/bin/python
RUN rm -f /usr/bin/python3 && ln -s /usr/bin/python3.8 /usr/bin/python3
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && rm get-pip.py
RUN pip3 install --upgrade pip && \
    # Print python an pip version
    python -V && pip3 -V
RUN pip3 install pyyaml
RUN pip3 install ipython


##############################################################################
# EFA Setup
##############################################################################
RUN cd ${STAGE_DIR} && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-1.17.2.tar.gz && \
    tar -xf aws-efa-installer-1.17.2.tar.gz && \
    cd aws-efa-installer && \
    ./efa_installer.sh -y -d -g  --skip-kmod --skip-limit-conf --no-verify


##############################################################################
# Some Packages
##############################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile-dev \
        libcupti-dev \
        libjpeg-dev \
        libpng-dev \
        screen automake

RUN pip3 install psutil \
                 yappi \
                 cffi \
                 ipdb \
                 pandas \
                 matplotlib \
                 py3nvml \
                 pyarrow \
                 graphviz \
                 astor \
                 boto3 \
                 tqdm \
                 sentencepiece \
                 msgpack \
                 requests \
                 pandas \
                 sphinx \
                 sphinx_rtd_theme \
                 scipy \
                 numpy \
                 sklearn \
                 scikit-learn \
                 nvidia-ml-py3 \
                 mpi4py \ 
                 cupy-cuda110 \
                 supervisor awscli



##############################################################################
# PyYAML build issue
# https://stackoverflow.com/a/53926898
##############################################################################
RUN rm -rf /usr/lib/python3/dist-packages/yaml && \
    rm -rf /usr/lib/python3/dist-packages/PyYAML-*

##############################################################################
# NCCL 2.10.3 Upgrade
##############################################################################
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" && apt update && apt install -y --allow-change-held-packages libnccl2=2.10.3-1+cuda11.0 libnccl-dev=2.10.3-1+cuda11.0
ENV NCCL_VERSION=2.12.12
RUN sudo apt install unzip
RUN cd ${STAGE_DIR} && \
    wget https://github.com/NVIDIA/nccl/archive/refs/tags/v${NCCL_VERSION}-1.zip && \
    unzip v${NCCL_VERSION}-1.zip && \
    cd ${STAGE_DIR}/nccl-${NCCL_VERSION}-1 && \
    make -j src.build && \
    make install

# USER deepspeed

# RUN echo "Current HOME=${HOME}}"

##############################################################################
# Install AWS-OFI-NCCL plugin
##############################################################################
RUN git clone https://github.com/aws/aws-ofi-nccl.git $STAGE_DIR/aws-ofi-nccl \
    && cd $STAGE_DIR/aws-ofi-nccl \
    && git checkout aws  \
    && ./autogen.sh \
    && ./configure --prefix=$STAGE_DIR/aws-ofi-nccl/install \
       --with-libfabric=/opt/amazon/efa/ \
       --with-cuda=/usr/local/cuda \
       --with-mpi=/opt/amazon/openmpi/ \
       --with-nccl=/usr/lib/x86_64-linux-gnu \
    && make -j$(nproc) && make install

ENV PATH=$PATH:/home/deepspeed/.local/bin
RUN echo "export PATH=$PATH:/home/deepspeed/.local/bin" >> /home/deepspeed/.bashrc
RUN hash -r
RUN pip3 install ninja pybind11 pyyaml mkl mkl-include cffi typing_extensions future six requests dataclasses h5py pytest-cov tensorboardX
# Extra installation
RUN pip3 install sentencepiece
RUN pip3 install -U pytest-cov setuptools py-cpuinfo

##############################################################################
# PyTorch
##############################################################################

RUN pip3 install --force-reinstall torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
    -f https://download.pytorch.org/whl/cu117/torch_stable.html
RUN pip3 install --no-deps pytorch-ignite 


##############################################################################
## Install custom Apex
##############################################################################
# for unknown reason the apex installation require USER deepspeed
USER deepspeed
RUN cd ${STAGE_DIR} && git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0" \
    pip3 install -v --disable-pip-version-check \
    --no-cache-dir --global-option="--cpp_ext" \
    --global-option="--cuda_ext" ./



##############################################################################
# DeepSpeed ZeRO2D code base
##############################################################################

#USER root
#ADD repo-key /
#RUN chmod 600 /repo-key && \
#    echo "IdentityFile /repo-key" >> /etc/ssh/ssh_config && \
#    echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config && \
#    echo "change code base: setting partition type from config" && \
#    git clone -b ms/debug git@github.com:zarzen/ZeRO-2D.git ${STAGE_DIR}/DeepSpeed && \
#    cd ${STAGE_DIR}/DeepSpeed \
#    && git submodule sync && git submodule update --init --recursive
#
#USER deepspeed
#RUN sudo chown $USER:$USER -R ${STAGE_DIR}/DeepSpeed
#RUN sudo apt-get install -y libaio-dev
#RUN cd ${STAGE_DIR}/DeepSpeed && \
#    sudo CUDACXX=/usr/local/cuda/bin/nvcc \
#    CUDA_HOME=/usr/local/cuda \
#    DS_BUILD_SPARSE_ATTN=0 \
#    DS_BUILD_OPS=1 TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0" pip3 install -v ./

# install nccl-tests
RUN git clone https://github.com/NVIDIA/nccl-tests.git $HOME/nccl-tests
RUN cd $HOME/nccl-tests \
    && make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/lib/x86_64-linux-gnu 

# Install Megatron-LM to $HOME
RUN pip3 install regex pynvml
RUN git clone https://github.com/NVIDIA/Megatron-LM.git $HOME/Megatron-LM
RUN cd $HOME/Megatron-LM && git checkout 0bb597b
ENV PYTHONPATH "${PYTHONPATH}:$HOME/Megatron-LM"

# Install xformers
RUN git clone https://github.com/jfc4050/xformers $HOME/xformers && \
    cd $HOME/xformers && git checkout -b bias-dropout origin/bias-dropout && \
    git checkout 500b8d4 && git submodule sync && git submodule update --init --recursive
RUN cd $HOME/xformers && \
    export FORCE_CUDA="1" && \
    export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0" && \
    CUDA_VISIBLE_DEVICES=0 pip3 install -e ".[dev]"

# Install epoi
RUN git clone https://github.com/comaniac/epoi --recursive $HOME/epoi
RUN cd $HOME/epoi && git fetch && git checkout b2e2e98 && pip3 install -e ".[dev]"

# Install transformers
RUN git clone https://github.com/huggingface/transformers.git $HOME/transformers
RUN cd $HOME/transformers && git checkout 2bdd9fa && pip3 install -e ".[dev]" --no-deps

# Fix dependencies
RUN pip3 install huggingface-hub tokenizers numpy==1.23.4

COPY container_entrypoint.sh /etc/
RUN sudo chown $USER:$USER /etc/container_entrypoint.sh
RUN chmod 777 /etc/container_entrypoint.sh
# ENTRYPOINT /etc/container_entrypoint.sh

# AWS Batch setup for CI
COPY batch_entry.sh /batch/entry.sh
RUN DEBIAN_FRONTEND=noninteractive sudo apt-get install -y awscli
RUN python3 -m pip install --upgrade awscli

EXPOSE 2345
EXPOSE 2555
EXPOSE 2022
