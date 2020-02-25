#!/bin/bash
cd "$(dirname $0)/.."
methods=(ResNet_DebugLL ResNet_LastBN_DebugLL)
depths=(56 110 230 1202)
seeds=(1)
batchSize=128
learningRate=0.1
weightDecay=0.0001
maxEpoch=160
eStep="{80,120}"
learningRateDecayRatio=0.1
l=${#methods[@]}
n=${#depths[@]}
f=${#seeds[@]}
for ((a=0;a<$l;++a))
do 
   for ((i=0;i<$n;++i))
   do 
      for ((b=0;b<$f;++b))
      do
    CUDA_VISIBLE_DEVICES=0 th main_Cifar10_DebugResNet.lua -model ${methods[$a]} -depth ${depths[$i]} -seed ${seeds[$b]} -batchSize ${batchSize} -learningRate ${learningRate} -weightDecay ${weightDecay}  -max_epoch ${maxEpoch} -learningRateDecayRatio ${learningRateDecayRatio} -epoch_step ${eStep}
      done
   done
done
