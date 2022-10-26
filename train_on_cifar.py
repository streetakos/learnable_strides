from typing import Any, Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
from torch import Tensor
import sys
sys.path.append("..") # Adds higher directory to python modules path.
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
from models.resnet import resnet18
from utils.helpers import seed_worker
from utils.helpers import print_params
from utils.helpers import set_cuda_seed
from utils.helpers import loss_function
from utils.helpers import load_config
from utils.helpers import Logger
from utils.helpers import save_config


# Parse Arguments and load Config
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--config', default=None, type=str, help='training config')
args = parser.parse_args()
config = load_config(args.config)
save_config(config, config["saving"]["save_path"],config["saving"]["name_of_logs"]+".yaml")


########## Hyperparams ################
seed = config["train"]["seed"]
device = torch.device("cuda:0")
batch_size = config["train"]["batch_size"]
num_of_epochs =  config["train"]["epochs"]
lr_strides = config["train"]["lr_strides"]
lr = config["train"]["lr"]
momentum = config["train"]["momentum"]
weight_decay =config["train"]["weight_decay"]

# Set seed for reproduction of results + CUDA seed
set_cuda_seed(seed)

# Import the CIFAR10 dataset
print('==> Preparing data..')
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_train = transforms.Compose([
    transforms.Resize(64),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.Resize(64),	
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
g = torch.Generator()
g.manual_seed(0)
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2, worker_init_fn=seed_worker, generator=g)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')




set_cuda_seed(seed)
net = resnet18().to(device)


# Define optimizer and Loss function
params = []
strides = [] 		
for name, param in net.named_parameters():
    if param.requires_grad:
        if 'stride' in name:			
            strides.append(param)
        else:
            params.append(param)			

if config["model"]["learn_strides"]:
    optimizer = optim.SGD([ {'params': iter(params)},
	  		{'params': iter(strides), 'lr': lr_strides, 'weight_decay': 0.0}
            ], lr=lr, momentum=momentum, weight_decay=weight_decay)
else:
    optimizer = optim.SGD(iter(params), lr=lr, weight_decay=weight_decay, momentum=momentum)  
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_of_epochs)
criterion = nn.CrossEntropyLoss()

# Define logger
fmt = {'kl': '3.3e',
       'tr_loss': '3.3e',
       'tst_acc': '.4f',
       'cnv_1_sx': '.4f',
	   'cnv_1_sy': '.4f',
	   'cnv_2_sx': '.4f',
	   'cnv_2_sy': '.4f',
	   'cnv_3_sx': '.4f',
	   'cnv_3_sy': '.4f',
	   'cnv_4_sx': '.4f',
	   'cnv_4_sy': '.4f',
       'te_acc_stoch': '.4f',
       'te_acc_ens10': '.4f',
       'te_acc_perm_sigma': '.4f',
       'te_acc_perm_sigma_ens': '.4f',
       'te_nll_ens100': '.4f',
       'te_nll_stoch': '.4f',
       'te_nll_ens10': '.4f',
       'exp_cal_er': '.4f',
       'te_nll_perm_sigma_ens': '.4f',
       'time': '.3f'}
logger = Logger(config["saving"]["name_of_logs"], fmt=fmt, base=config["saving"]["save_path"])
logger.base = config["saving"]["save_path"]	

# Training the neural network
print('==> Start Training ..')

for epoch in range(num_of_epochs):

    running_loss = 0.0
    steps_per_epoch = 0	
    for i, data in enumerate(trainloader, 0):
      
        inputs, labels =  data[0].to(device), data[1].to(device)		
		
        # Fix the stride parameters after some epochs		
        if config["model"]["fix_strides_after_some_epochs"] and epoch == config["model"]["fix_on_epochs"]:		
            params = []
            strides = [] 		
            for name, param in net.named_parameters():
                if param.requires_grad:
                    if 'stride' in name:	
                        print(name)						
                        param.requires_grad = False
                    else:
                        params.append(param)
            torch.save(net.state_dict(), os.path.join(config["saving"]["save_path"],"model_at_70_epochs.pt") )								
            optimizer = optim.SGD(iter(params), lr=lr, weight_decay=weight_decay)  

        if (inputs.shape[0] % min(inputs.shape[0],32) ) == 0 :

            steps_per_epoch +=1			
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            #loss = loss_function(outputs, labels,net)			
            loss.backward()
			
            #print(  net.conv1.weight.grad.detach().cpu().numpy()[0,0,0,0:5]  )	
            #print(  net.conv1.stride.grad.detach().cpu().numpy()  )	
            #grads = torch.autograd.grad(loss, net.parameters())
            #print(len(grads))	
            #print(  net.conv1.weight.grad.detach().cpu().numpy()[0,0,0,0:5]  )		
			
            #print(steps_per_epoch,net.conv1.stride.grad.detach().cpu().numpy())			
            optimizer.step()
            running_loss += loss.item()


            
    			
    # Display the test accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(testloader,0):			
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    	
		
	#Save strides	
    if config["model"]["learn_strides"]:
        strds = []	
        for name, module in net.named_modules():
            if isinstance(module, StrideConv2d):
                strds.append(module.stride.detach().cpu().numpy()[0])
                strds.append(module.stride.detach().cpu().numpy()[1])
				
        logger.add(epoch, cnv_1_sx= strds[0], cnv_1_sy= strds[1])
        logger.add(epoch, cnv_2_sx= strds[2], cnv_2_sy= strds[3])
        logger.add(epoch, cnv_3_sx= strds[4], cnv_3_sy= strds[5])
        logger.add(epoch, cnv_4_sx= strds[6], cnv_4_sy= strds[7])
	
    logger.add(epoch, tr_loss=running_loss/steps_per_epoch, tst_acc=(100.0 * correct / total)) 	
    logger.iter_info()
    logger.save(silent=True)			
    steps_per_epoch = 0
    running_loss = 0.0	
	
    scheduler.step() 
	

print('Finished Training')
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
torch.save(net.state_dict(), os.path.join(config["saving"]["save_path"],"mossaaasfadel.pt") )








				