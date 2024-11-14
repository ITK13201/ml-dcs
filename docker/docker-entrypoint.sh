#!/usr/bin/env sh

set -eu

APP=ml-dcs
ANACONDA_BIN=/opt/conda/bin
EXEC_JUPYTER_LAB=${ANACONDA_BIN}/jupyter-lab
EXEC_PYTHON=${ANACONDA_BIN}/python

usage() {
    echo "Usage: $0 [-d] [-q]" 1>&2
    echo "Options: " 1>&2
    echo "-d: Run as development mode" 1>&2
    echo "-q: Quit" 1>&2
    exit 1
}

ENVIRONMENT=prod
QUIT=0

while getopts :dqh OPT
do
    case $OPT in
    d)  ENVIRONMENT=dev
        ;;
    q)  QUIT=1
        ;;
    h)  usage
        ;;
    \?) usage
        ;;
    esac
done

if [ "$QUIT" = "1" ]; then
    exit 0
fi

if [ "${ENVIRONMENT:-}" = "dev" ]; then
    ${EXEC_JUPYTER_LAB} --ip=0.0.0.0 --port="${PORT:-8000}" --no-browser --NotebookApp.token='token'
elif [ "${ENVIRONMENT:-}" = "prod" ]; then
    ${EXEC_PYTHON} /usr/local/src/"${APP}"/main.py
fi
