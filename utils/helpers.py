from src.strides import StrideConv2d
from src.strides import Conv2d_StridesAsInput
from src.strides import Conv2dStride2
import numpy as np
import random
import torch.nn.functional as F
import os
import argparse
import torch
from torch import Tensor
from torch import nn
from src.strides import StrideConv2d
import torch.nn.functional as F
#from utils.logger import Logger
import torch
import torchvision
import torchvision.transforms as transforms
#from the_vgg import vgg11,vgg11_bn
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch.optim import Adam
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import warnings
from src.strides import Conv2dStride2
import yaml

import os
import sys
import random
import numpy as np

from collections import OrderedDict
from tabulate import tabulate
from pandas import DataFrame
from time import gmtime, strftime


def seed_worker(worker_id):
    worker_seed = 127
    np.random.seed(worker_seed)
    random.seed(worker_seed)
	
def print_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.data)
		
def set_cuda_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True	
    warnings.warn('You have chosen to seed training. '
			  'This will turn on the CUDNN deterministic setting, '
			  'which can slow down your training considerably! '
			  'You may see unexpected behavior when restarting '
			  'from checkpoints.')
	
def loss_function(x, x_hat, net):
    criterion = nn.CrossEntropyLoss()	
    reproduction_loss = criterion(x, x_hat)
    
    # Gather strides	
    strds = []	
    for name, module in net.named_modules():
        if isinstance(module, StrideConv2d):
            strds.append(module.stride[0])
            strds.append(module.stride[1])
			
    prior_loss = 0		
    for s in strds:
        param = 1.0/(torch.exp(s)+1.0)
        sigma_hat = param**2
        prior_loss += 0.5*torch.log(sigma_hat) + ( param**2 / 2.0*sigma_hat)

		
    return reproduction_loss + 0.0001 * prior_loss	 #0.01

def load_config(config_name):
    #with open(os.path.join(CONFIG_PATH, config_name)) as file:
    with open( config_name ) as file:
		
        config = yaml.safe_load(file)

    return config

def save_config(config,path,config_name):
    os.makedirs(path, exist_ok=True)	
    with open(  os.path.join(path ,config_name), 'w') as file:
        documents = yaml.dump(config, file)
		

class Logger:
    def __init__(self, name='name', fmt=None, base = './logs'):
        self.handler = True
        self.scalar_metrics = OrderedDict()
        self.fmt = fmt if fmt else dict()

        self.base = base
        if not os.path.exists(self.base): os.mkdir(self.base)

        time = gmtime()
        hash = ''.join([chr(random.randint(97, 122)) for _ in range(5)])
        fname = '-'.join(sys.argv[0].split('/')[-3:])
        self.path = '%s/%s-%s-%s-%s' % (self.base, fname, name, strftime('%m-%d-%H-%M', time), hash)

        self.logs = self.path + '.csv'
        self.output = self.path + '.out'
        self.checkpoint = self.path + '.cpt'

        def prin(*args):
            str_to_write = ' '.join(map(str, args))
            with open(self.output, 'a') as f:
                f.write(str_to_write + '\n')
                f.flush()

            print(str_to_write)
            sys.stdout.flush()

        self.print = prin

    def add_scalar(self, t, key, value):
        if key not in self.scalar_metrics:
            self.scalar_metrics[key] = []
        self.scalar_metrics[key] += [(t, value)]

    def add_dict(self, t, d):
        for key, value in d.iteritems():
            self.add_scalar(t, key, value)

    def add(self, t, **args):
        for key, value in args.items():
            self.add_scalar(t, key, value)

    def iter_info(self, order=None):
        names = list(self.scalar_metrics.keys())
        if order:
            names = order
        values = [self.scalar_metrics[name][-1][1] for name in names]
        t = int(np.max([self.scalar_metrics[name][-1][0] for name in names]))
        fmt = ['%s'] + [self.fmt[name] if name in self.fmt else '.1f' for name in names]

        if self.handler:
            self.handler = False
            self.print(tabulate([[t] + values], ['epoch'] + names, floatfmt=fmt))
        else:
            self.print(tabulate([[t] + values], ['epoch'] + names, tablefmt='plain', floatfmt=fmt).split('\n')[1])

    def save(self, silent=False):
        result = None
        for key in self.scalar_metrics.keys():
            if result is None:
                result = DataFrame(self.scalar_metrics[key], columns=['t', key]).set_index('t')
            else:
                df = DataFrame(self.scalar_metrics[key], columns=['t', key]).set_index('t')
                result = result.join(df, how='outer')
        result.to_csv(self.logs)
        if not silent:
            self.print('The log/output/model have been saved to: ' + self.path + ' + .csv/.out/.cpt')