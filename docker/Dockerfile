FROM continuumio/anaconda3:2024.06-1

ARG APP=ml-dcs

WORKDIR /usr/local/src/${APP}

RUN apt-get update -y \
    && conda config --set always_yes yes \
    && conda update -y conda

RUN useradd --create-home --shell /bin/bash ${APP}
USER ${APP}

COPY --chown=${APP}:${APP} ./${APP}/environment.yaml ./environment.yaml

RUN conda env create -n ${APP} -f ./environment.yaml \
    && conda init \
    && echo ". activate ${APP}" >> /home/${APP}/.bashrc

COPY --chown=${APP}:${APP} ./${APP} .

COPY --chown=${APP}:${APP} ./docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["-d"]
