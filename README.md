# ML-DCS (Machine Learning for Discrete Controller Synthesis)

## Dependencies

- Docker
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
  - [How to install](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Caution

Restart docker after installing the NVIDIA Container Toolkit.

## Confirm of installation of NVIDIA Container Toolkit

Add the following items to compose.yaml and execute.
nvidia-smi is executed.

```yaml
services:
  anaconda:
    command:
      - -g
    ...
```
