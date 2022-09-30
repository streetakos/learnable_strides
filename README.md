# Learnable Strides in Spatial Domain


This code is based upon [mmcv](https://github.com/open-mmlab/mmcv/tree/master/mmcv/ops) implementation of Deformable Convolutional Networks

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
Move to `src/` folder and execute the following
```
python setup.py install
```

See `test.py` for example usage.


## TODOs

- [x] Forward Pass
- [x] Backward Pass
- [x] Gradiets of operations with integer strides match the standard results
- [x] Add Biases to the Conv Layer
- [ ] Check the gradients of strides
- [ ] Account for batches in strides of gradient (I think pytorch does this automatically)
