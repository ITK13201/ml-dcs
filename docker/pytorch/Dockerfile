FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS development

ARG APP=ml-dcs
WORKDIR /usr/local/src/${APP}

RUN apt-get update -y \
    && apt-get install -y \
        sudo \
        wget \
        vim \
        curl \
        less

# Install libraries
RUN conda install -y \
        conda-forge::jupyterlab \
        conda-forge::pydantic \
        conda-forge::scipy \
        conda-forge::matplotlib \
        conda-forge::pandas \
        conda-forge::scikit-learn
RUN pip install \
        torch_geometric

# Create user
ARG PASSWORD=password
ARG GID=1000
ARG UID=1000
RUN groupadd -g ${GID} ${APP} \
    && useradd --create-home --shell /bin/bash -u ${UID} -g ${GID} -G sudo ${APP} \
    && echo ${APP}:${PASSWORD} | chpasswd \
    && echo "${APP}   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER ${APP}

# Copy docker-entrypoint.sh
COPY --chown=${APP}:${APP} ./docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy source files
COPY --chown=${APP}:${APP} ./${APP} .

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["-d"]


FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime AS production

ARG APP=ml-dcs
WORKDIR /usr/local/src/${APP}

RUN apt-get update -y \
    && apt-get install -y \
        sudo \
        wget \
        vim \
        curl \
        less

# Install libraries
RUN conda install -y \
        conda-forge::jupyterlab \
        conda-forge::pydantic \
        conda-forge::scipy \
        conda-forge::matplotlib \
        conda-forge::pandas \
        conda-forge::scikit-learn
RUN pip install \
        torch_geometric

# Create user
ARG PASSWORD=password
ARG GID=1000
ARG UID=1000
RUN groupadd -g ${GID} ${APP} \
    && useradd --create-home --shell /bin/bash -u ${UID} -g ${GID} -G sudo ${APP} \
    && echo ${APP}:${PASSWORD} | chpasswd \
    && echo "${APP}   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER ${APP}

# Copy docker-entrypoint.sh
COPY --chown=${APP}:${APP} ./docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy source files
COPY --chown=${APP}:${APP} ./${APP} .

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD []
