FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

MAINTAINER Boris Oreshkin <boris@elementai.com>

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --no-check-certificate --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

RUN apt-get update && apt-get install -y \
    libhdf5-dev

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /opt/conda/bin:$PATH

RUN conda install python=3.6.7

RUN mkdir -p -m 700 /root/.jupyter/ && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = 'sha1:3f84353ad3f5:d1b6eeb440acbc49330646714898ae27c8dd56c2'" >> /root/.jupyter/jupyter_notebook_config.py

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

RUN conda install -y pymongo pandas jupyter pillow requests matplotlib tqdm \
    h5py pytorch=0.4.1 opencv

RUN pip install pyyaml celery tensorflow-gpu==1.8.0 tensorboard==1.8.0 statsmodels gitpython \ 
    scikit-image opencv-python scikit-learn googledrivedownloader scipy==0.19.0 torchvision==0.2.0 \
    numpy==1.15.3 tensorboardX torch==0.4.1

ENTRYPOINT [ "/usr/bin/tini", "--" ]

CMD ["jupyter", "notebook", "--allow-root"]
