# Learnable Strides in Spatial Domain


This code is based upon [mmcv](https://github.com/open-mmlab/mmcv/tree/master/mmcv/ops) implementation of Deformable Convolutional Networks and the following [repo] (https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch)

## Requirements and Current Setup
The complete enviroment for building the code:
- ninja
- pytorch = '1.7.1+cu110'
- setuptools
- cuda = 11.0

## Build
Before building the source code or activate any virtual enviroment be sure that the path
points to the right version of the installed cuda
```
export PATH=/home/streetakos/.local/bin:/usr/local/cuda-11.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/local/cuda-11.0/bin:/home/streetakos/bin:/home/streetakos/bin:/home/streetakos/bin
```
Move to `src/` and then to `src/stride_input_conv/` folder and execute the following:
```
python setup.py install
```
See `test.py` for example usage.

## Training 
In order to train the moel to the Cifar10 dataset specify the training prameters in the config file and the execute:
```
python3 train_on_cifar.py --config "config.yaml"
```

## TODOs

- [x] Forward Pass
- [x] Backward Pass
- [x] Gradiets of operations with integer strides match the standard results
- [x] Add Biases to the Conv Layer
- [ ] Check the gradients of strides
- [x] Account for batches in strides of gradient (I think pytorch does this automatically)
- [x] Learn only positive strides (log_strides ?)

