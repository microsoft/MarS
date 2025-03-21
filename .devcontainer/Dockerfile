FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN nvcc --version && \
    echo "CUDA Version: $(cat /usr/local/cuda/version.txt 2>/dev/null || echo 'Not found')"

WORKDIR /workspaces
RUN chmod -R a+w /workspaces

# Set timezone consistently
ENV TZ=Asia/Chongqing
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN DEBIAN_FRONTEND=noninteractive && \
    apt-get -y update && \
    apt-get -y install --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
        tmux \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN ( \
    cd && \
    git clone https://github.com/gpakosz/.tmux.git && \
    ln -s -f .tmux/.tmux.conf && \
    cp .tmux/.tmux.conf.local . \
)

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install Miniconda and Python packages
RUN curl -o miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda install -y python=3.12 conda-build && \
    /opt/conda/bin/conda clean -ya

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

RUN python -m pip list --format=freeze
