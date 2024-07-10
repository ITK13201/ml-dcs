#!/usr/bin/env sh

set -eu

APP=ml-dcs

usage() {
    echo "Usage: $0 [-d] [-g] [-q]" 1>&2
    echo "Options: " 1>&2
    echo "-d: Run as development mode" 1>&2
    echo "-g: Run nvidia smi"
    echo "-q: Quit without running server" 1>&2
    exit 1
}

ENVIRONMENT=prod
NVIDIA_SMI=0
QUIT=0

while getopts :dgqh OPT
do
    case $OPT in
    d)  ENVIRONMENT=dev
        ;;
    g)  NVIDIA_SMI=1
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

if [ "$NVIDIA_SMI" = "1" ]; then
    nvidia-smi
elif [ "${ENVIRONMENT:-}" = "dev" ]; then
    jupyter-lab --ip=0.0.0.0 --port="${PORT:-8000}" --no-browser --NotebookApp.token=''
elif [ "${ENVIRONMENT:-}" = "prod" ]; then
    python /usr/local/src/"${APP}"/main.py
fi
