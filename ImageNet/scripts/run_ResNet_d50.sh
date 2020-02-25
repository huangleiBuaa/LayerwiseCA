#!/bin/bash
cd "$(dirname $0)/.."
CUDA_VISIBLE_DEVICES=0,1,2,3 th main_ImageNet.lua -model ResNet -depth 50  -batchSize 256 -LR 0.1 -weightDecay 0.0001 -nEpochs 100 -nDecayEpoch 30 -nGPU 4
