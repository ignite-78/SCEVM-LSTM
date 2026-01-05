# Synergizing vision mamba and spatial channel reconstruction convolutional attention for radar-based precipitation nowcasting

Official repository for SCEVM-LSTM: Synergizing vision mamba and spatial channel reconstruction convolutional attention for radar-based precipitation nowcasting

![Example Image](figures/SCEVM-LSTM.png)

## Introduction

We propose an innovative spatiotemporal long short-term memory model (SCEVM-LSTM), which merges Vision Mamba blocks with spatial and channel reconstruction convolutional modules incorporating efficient multi-scale attention (SCEMA). The experiments demonstrate that this model significantly enhances the prediction capability for high echoes and better captures long-term trends in precipitation systems, thereby improving the model's performance under extended time steps.

## Installation

```
pip install causal_conv1d==1.1.1
pip install mamba_ssm==1.1.1
```
Special note: Different versions of mamba-ssm can still run our code, but their versions must match those of Python, PyTorch, and CUDA! The vast majority of mamba environment configuration issues stem from mismatched CUDA versions!

## Datasets
All the two datasets in our paper is publicly available. You can find the datasets as follows:
- [CIKM](https://tianchi.aliyun.com/dataset/1085)
- [HKO-7](https://github.com/sxjscience/HKO-7)

## Overview

- `data/:` contains CIKM.(Please put the processed CIKM dataset in this folder.)
- `openstl/methods/scevm_lstm.py:` contains defined training method of SCEVM-LSTM.
- `openstl/models/scevm_lstm_model.py:` contains the model SCEVM-LSTM.
- `configs` contains training configs for CIKM.

## Train

### CIKM

```
python3 train.py
```

## Acknowledgments

Our code is based on [OpenSTL](https://github.com/chengtan9907/OpenSTL) and [VMRNN](https://github.com/yyyujintang/VMRNN-PyTorch). We sincerely appreciate for their contributions.

