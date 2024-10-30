FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

ARG APP=ml-dcs
WORKDIR /usr/local/src/${APP}

RUN apt-get update -y \
    && apt-get install -y \
        sudo \
        wget \
        vim \
        curl

# Install anaconda
ARG ANACONDA_INSTALLER_VERSION=2024.10-1
RUN curl -O https://repo.anaconda.com/archive/Anaconda3-${ANACONDA_INSTALLER_VERSION}-Linux-x86_64.sh \
    && bash ./Anaconda3-${ANACONDA_INSTALLER_VERSION}-Linux-x86_64.sh -b -p /opt/conda \
    && rm -f ./Anaconda3-${ANACONDA_INSTALLER_VERSION}-Linux-x86_64.sh
ENV PATH /opt/conda/bin:${PATH}
RUN conda update -y conda

# Create user
RUN useradd --create-home --shell /bin/bash ${APP}
USER ${APP}

# Install python libraries
COPY --chown=${APP}:${APP} ./${APP}/environment.yaml ./environment.yaml
RUN conda env create -n ${APP} -f ./environment.yaml \
    && conda init \
    && echo ". activate ${APP}" >> /home/${APP}/.bashrc
ENV CONDA_DEFAULT_ENV ${APP}
ENV PATH /opt/conda/envs/${APP}/bin:${PATH}

# Copy docker-entrypoint.sh
COPY --chown=${APP}:${APP} ./docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy source files
COPY --chown=${APP}:${APP} ./${APP} .

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["-d"]
