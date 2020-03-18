This code is for the following paper: 

**Layer-wise Conditioning  Analysis in Exploring the Learning Dynamics of DNNs**

Lei Huang, Dawei Yang, Bo Lang, Jia Deng.  [arXiv:2002.10801](https://arxiv.org/abs/2002.10801)

================================================================================
This implementation is based on Torch. The experiments in the paper are run on cuda9.0 and cudnn 7.0.

(1) We wrap the implementation of the conditioning analysis in the linear layer (./Cifar/module/Linear_AllMode.lua) and convolution layer (./Cifar/module/SpatialConvolution_conditioning.lua)



(2) We provide the scripts to reproduce the experimental results, described in Section 5 of the paper.


To reproduce the results, you need:

1. install Torch (http://torch.ch/), with CUDA and cudnn.  install the dependency optnet by: luarocks install optnet



2. Reproduce experiments on CIFAR datasets: 
   (1) prepare dataset: download CIFAR-10 (https://yadi.sk/d/eFmOduZyxaBrT) and Cifar-100 dataset (https://yadi.sk/d/ZbiXAegjxaBcM), and put the data files under `./Cifar/data/'
   
    (2) Reproduce the ill-conditioned problem of resiudal network: 
        `cd   Cifar/scripts/'
        run the scripts:`bash  run_bash_Cifar10_DebugResNet.sh'
   
    (3) Reproduce the performances on CIFAR-10: 
        `cd   Cifar/scripts/'
        run the scripts:`bash  run_bash_Cifar10.sh'
    
    (4) Reproduce the performances on CIFAR-100: 
        `cd   Cifar/scripts/'
        run the scripts:`bash  run_bash_Cifar100.sh'


2. Reproduce experiments on ImageNet: 
   (1) prepare dataset: download ImageNet-2012 and put the dataset under `./ImageNet/data/'

   (2) run all the scripts in `./ImageNet/scripts/'


   Note that: the 1202 ResNet for Cifar-10 experiments, and the imageNet experiments require GPUs with memory larger than 26 GB(we run the experiments on NVIDIA V100.).
   
   This code is based on the facebook's github project for residual network: 
       https://github.com/facebook/fb.resnet.torch 
