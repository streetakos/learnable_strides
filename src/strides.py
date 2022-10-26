# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single
from torch.nn import init
import numpy as np
import random
import stride_conv_cuda
import stride_input_conv_cuda_deterministic

class Conv2d_StridesAsInput(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 groups: int = 1,
                 bias: bool = True) -> None:
        super(Conv2d_StridesAsInput, self).__init__()

        assert in_channels % groups == 0, \
            f'in_channels {in_channels} cannot be divisible by groups {groups}'
        assert out_channels % groups == 0, \
            f'out_channels {out_channels} cannot be divisible by groups \
              {groups}'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.transposed = False
        self.output_padding = _single(0)
        self.use_bias = bias
		
        # Define Learnable parameters
        self.weight = nn.Parameter( torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        
        # If the use_bias is true the the bias will be zero for all the training (I assume) 		
        self.bias = nn.Parameter(torch.zeros(out_channels) )
         		
        # Initialize parameters		
        self.reset_parameters()

        if self.use_bias is False :
            self.bias.requires_grad = False
			
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
		
        if self.use_bias is True:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)		

    def forward(self, x: Tensor, stride_h: float, stride_w: float) -> Tensor:
        input_pad = (x.size(2) < self.kernel_size[0]) or (x.size(3) < self.kernel_size[1])
        
        # Add zero pad if needed		
        if input_pad:
            pad_h = max(self.kernel_size[0] - x.size(2), 0)
            pad_w = max(self.kernel_size[1] - x.size(3), 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
        
		
		
        out = stride_conv2d_as_input(x, self.weight, stride_h, stride_w, self.bias, self.padding,
                            self.dilation, self.groups)
		

		
        if input_pad:
            out = out[:, :, :out.size(2) - pad_h, :out.size(3) -
                      pad_w].contiguous()
        return out

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(in_channels={self.in_channels}, '
        s += f'out_channels={self.out_channels}, '
        s += f'kernel_size={self.kernel_size}, '
        s += f'padding={self.padding}, '
        s += f'dilation={self.dilation}, '
        s += f'groups={self.groups}, '
        s += f'bias={self.use_bias})'
        return s	

class StrideConv2d(nn.Module):
    r"""Strided 2D convolution.
    Applies a 2D convolution over an input signal composed of
    several input planes.
	
    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size(int, tuple): Size of the convolving kernel.
        padding (int or tuple): Zero-padding added to both sides of the input. Default: 0.
        dilation (int or tuple): Spacing between kernel elements. Default: 1.
        groups (int): Number of blocked connections from input. channels to output channels. Default: 1.
        bias (bool): If True, adds a learnable bias to the output. Default: False.

    """

    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 groups: int = 1,
                 bias: bool = True) -> None:
        super(StrideConv2d, self).__init__()

        assert in_channels % groups == 0, \
            f'in_channels {in_channels} cannot be divisible by groups {groups}'
        assert out_channels % groups == 0, \
            f'out_channels {out_channels} cannot be divisible by groups \
              {groups}'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.transposed = False
        self.output_padding = _single(0)
        self.use_bias = bias
		
        # Define Learnable parameters
        self.weight = nn.Parameter( torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        # TODO define only strides >= 1 with log(strides) + 1
        self.stride = nn.Parameter(torch.ones(2) +1.0) 
        #self.stride = nn.Parameter(torch.zeros(2)) 		
        
        # If the use_bias is true the the bias will be zero for all the training (I assume) 		
        self.bias = nn.Parameter(torch.zeros(out_channels) )
         		
        # Initialize parameters		
        self.reset_parameters()
        #print(self.stride)
        if self.use_bias is False :
            self.bias.requires_grad = False
			
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
		
        if self.use_bias is True:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)		

    def forward(self, x: Tensor) -> Tensor:
        """Stride Convolutional forward function.

        Args:
            x (Tensor): Input feature, shape (B, C_in, H_in, W_in)
        Returns:
            Tensor: Output of the layer.
        """
		
		
        '''		
        # Bound strides to (0,H]
        with torch.no_grad():	
            high_h =  x.shape[2] 
            high_w =  x.shape[3] 			
			
            r_strs = torch.exp(self.stride) + 1.0		
		
            h_h = 1.0 - (r_strs[0]>=high_h).float()	 	
            h_w = 1.0 - (r_strs[1]>=high_w).float()	
		
            if high_h > 1 and high_w > 1:		
                self.stride[0] = h_h*self.stride[0] + (1.0 - h_h)* (np.log( high_h-1.0 ))				
                self.stride[1] = h_w*self.stride[1] + (1.0 - h_w)* (np.log( high_w-1.0 ))
            # In case of stride 1 the learned stride has to get value close to zero 				
            if high_h == 1:
                self.stride[0] = h_h*self.stride[0] + (1.0 - h_h)* (np.log( 0.1 ))				
            if high_w == 1:
                self.stride[1] = h_h*self.stride[1] + (1.0 - h_h)* (np.log( 0.1 ))
				
            #print(high_h,high_w, torch.exp(self.stride)+1.0)
			
        '''	
        #with torch.no_grad():	
        #    self.stride[0] = torch.floor(self.stride[0] )
        #    self.stride[1] = torch.floor(self.stride[1] )
					
		
        #print(self.stride)		
        with torch.no_grad():	
			
            low = 1.0 
            #high_h =  input.shape[2]/3
            #high_w =  input.shape[3]/3		

            c_h = 1.0 - (self.stride[0]<low).float() 
            c_w = 1.0 - (self.stride[1]<low).float()
		 	
            #h_h = 1.0 - (stride[0]>high_h).float()	 	
            #h_w = 1.0 - (stride[1]>high_w).float()	
		
            self.stride[0] = c_h*self.stride[0] + (1.0 - c_h)*1.0				
            self.stride[1] = c_w*self.stride[1] + (1.0 - c_w)*1.0		
		
            #stride[0] = h_h*stride[0] + (1.0 - h_h)* (high_h*1.0)				
            #stride[1] = h_w*stride[1] + (1.0 - h_w)* (high_w*1.0)
		


        # To fix an assert error in deform_conv_cuda.cpp:128
        # input image is smaller than kernel
        input_pad = (x.size(2) < self.kernel_size[0]) or (x.size(3) < self.kernel_size[1])
		
        # Add zero pad if needed		
        if input_pad:
            pad_h = max(self.kernel_size[0] - x.size(2), 0)
            pad_w = max(self.kernel_size[1] - x.size(3), 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
        
		
        #true_stride = torch.exp(self.stride) + 1.0		
		
		
        # The computational part is done in the specific function
        out = stride_conv2d(x, self.weight, self.stride, self.bias, self.padding,
                            self.dilation, self.groups)

        if input_pad:
            out = out[:, :, :out.size(2) - pad_h, :out.size(3) -
                      pad_w].contiguous()
			
        #print(high_h,high_w, out.shape, torch.exp(self.stride)+1.0)			
        return out

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(in_channels={self.in_channels}, '
        s += f'out_channels={self.out_channels}, '
        s += f'kernel_size={self.kernel_size}, '
        s += f'padding={self.padding}, '
        s += f'dilation={self.dilation}, '
        s += f'groups={self.groups}, '
        s += f'bias={self.use_bias})'
        return s		

class Conv2dStride2(nn.Module):
    r"""Strided 2D convolution.
    Applies a 2D convolution over an input signal composed of
    several input planes.
	
    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size(int, tuple): Size of the convolving kernel.
        padding (int or tuple): Zero-padding added to both sides of the input. Default: 0.
        dilation (int or tuple): Spacing between kernel elements. Default: 1.
        groups (int): Number of blocked connections from input. channels to output channels. Default: 1.
        bias (bool): If True, adds a learnable bias to the output. Default: False.

    """

    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 groups: int = 1,
                 bias: bool = True) -> None:
        super(Conv2dStride2, self).__init__()

        assert in_channels % groups == 0, \
            f'in_channels {in_channels} cannot be divisible by groups {groups}'
        assert out_channels % groups == 0, \
            f'out_channels {out_channels} cannot be divisible by groups \
              {groups}'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.transposed = False
        self.output_padding = _single(0)
        self.use_bias = bias
		
        # Define Learnable parameters
        self.weight = nn.Parameter( torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.stride = nn.Parameter(torch.ones(2)+1.0, requires_grad = False) 		
        
        # If the use_bias is true the the bias will be zero for all the training (I assume) 		
        self.bias = nn.Parameter(torch.zeros(out_channels) )
         		
        # Initialize parameters		
        self.reset_parameters()

        if self.use_bias is False :
            self.bias.requires_grad = False
			
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
		
        if self.use_bias is True:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)		

    def forward(self, x: Tensor) -> Tensor:
        """Stride Convolutional forward function.

        Args:
            x (Tensor): Input feature, shape (B, C_in, H_in, W_in)
        Returns:
            Tensor: Output of the layer.
        """
		
        # To fix an assert error in deform_conv_cuda.cpp:128
        # input image is smaller than kernel
        input_pad = (x.size(2) < self.kernel_size[0]) or (x.size(3) < self.kernel_size[1])
        
        # Add zero pad if needed		
        if input_pad:
            pad_h = max(self.kernel_size[0] - x.size(2), 0)
            pad_w = max(self.kernel_size[1] - x.size(3), 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
        
        # The computational part is done in the specific function
        #stride = torch.log(self.stride+1.718282)		
        #print('Hey') 		
        out = stride_conv2d(x, self.weight, self.stride, self.bias, self.padding,
                            self.dilation, self.groups)
        if input_pad:
            out = out[:, :, :out.size(2) - pad_h, :out.size(3) -
                      pad_w].contiguous()
        return out

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(in_channels={self.in_channels}, '
        s += f'out_channels={self.out_channels}, '
        s += f'kernel_size={self.kernel_size}, '
        s += f'padding={self.padding}, '
        s += f'dilation={self.dilation}, '
        s += f'groups={self.groups}, '
        s += f'bias={self.use_bias})'
        return s			
	
class StrideConv2dFunction(Function):

    @staticmethod
    def forward(ctx,
                input,
                weight,
                stride,
				bias,
                padding=0,
                dilation=1,
                groups=1,
                im2col_step=32):
		
        if input is not None and input.dim() != 4:
            raise ValueError(
                f'Expected 4D tensor as input, got {input.dim()}D tensor \
                  instead.')
        #assert bias is False, 'Only support bias is False.'
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.im2col_step = im2col_step

        # Asert on strides <1 works		
        #stride = stride*0.0 +0.5 #torch.nn.Parameter(	torch.ones(2) - 0.5 )
        #stride = stride + 1.0 #torch.nn.Parameter(	torch.ones(2) - 0.5 )		
        		
        input = input.type_as(stride)
        weight = weight.type_as(input)
		
		
        #Bound strides	##################################################
        '''		
        low = 1.0 
        high_h =  input.shape[2]/3
        high_w =  input.shape[3]/3		
        #self.high = 1 		
        c_h = 1.0 - (stride[0]<low).float() 
        c_w = 1.0 - (stride[1]<low).float()
		 	
        h_h = 1.0 - (stride[0]>high_h).float()	 	
        h_w = 1.0 - (stride[1]>high_w).float()	
		
        stride[0] = c_h*stride[0] + (1.0 -c_h)*1.0				
        stride[1] = c_w*stride[1] + (1.0 - c_w)*1.0		
		
        stride[0] = h_h*stride[0] + (1.0 - h_h)* (high_h*1.0)				
        stride[1] = h_w*stride[1] + (1.0 - h_w)* (high_w*1.0)
        '''	
        
		
        '''
        #Working Bound
        high_h =  input.shape[2]#-1
        high_w =  input.shape[3]#-1
		
        #self.high = 1
        r_strs = torch.exp(stride) + 1.0		
		
        h_h = 1.0 - (r_strs[0]>=high_h).float()	 	
        h_w = 1.0 - (r_strs[1]>=high_w).float()	
		
        stride[0] = h_h*stride[0] + (1.0 - h_h)* (np.log( high_h-1.0 ))				
        stride[1] = h_w*stride[1] + (1.0 - h_w)* (np.log( high_w-1.0 ))				
        #print(input.shape, r_strs, high_h,high_w, torch.exp(stride) + 1.0)  		
        '''		
		
        ctx.save_for_backward(input, weight, stride, bias)
        
        #stride = 
        # Compute the Output shape
        # code for log strides
        #lsh = int(torch.floor( torch.exp(stride)+1  )[0])
        #lsw = int(torch.floor( torch.exp(stride)+1  )[1])
        #int_strides = ( lsh,lsw ) ###########################################
        ''' 		
        #int_strides = ( int(torch.floor(stride)[0]),int(torch.floor(stride)[1]) )		
        sf = StrideConv2dFunction._output_size(ctx, input, weight, int_strides)
        oh = sf[2]
        ow = sf[3]			
        if sf[2] == 2:
            print(int_strides)			
            oh = 2
            stride[0] = torch.log( torch.exp(stride[0]) - 1.0  )
        if sf[3] == 2:		
            print(int_strides)			
            ow = 2
            stride[1] = torch.log( torch.exp(stride[1]) - 1.0  )	
        lsh = int(torch.floor( torch.exp(stride)+1  )[0])
        lsw = int(torch.floor( torch.exp(stride)+1  )[1])
        int_strides = ( lsh,lsw )			
        shaper = (sf[0],sf[1],oh,ow)	
        '''		
        int_strides = ( int(torch.floor(stride)[0]),int(torch.floor(stride)[1]) )
        #int_strides = ( int(stride[0]),int(stride[1]) )		
        #print(stride, int_strides, int(stride[0].item()),int(stride[1].item()) )		
        output = input.new_empty(
            StrideConv2dFunction._output_size(ctx, input, weight, int_strides))
        #print(output.shape)		
        '''
        if	output.shape[2]   < weight.shape[2] :
            r_strs = torch.exp(stride) 		
            stride[0] = torch.log(torch.exp(stride[0])-0.1)
            int_strides = ( int(torch.floor( torch.exp(stride)+1  )[0]),int(torch.floor( torch.exp(stride)+1 )[1]) )
            output = input.new_empty(
                StrideConv2dFunction._output_size(ctx, input, weight, int_strides))
            print("THIS IS ERROR ",torch.exp(stride) )
			
        if	output.shape[3]   < weight.shape[3]:
            r_strs = torch.exp(stride) 		
            stride[1] = torch.log(torch.exp(stride[1])-0.1)
            int_strides = ( int(torch.floor( torch.exp(stride)+1  )[0]),int(torch.floor( torch.exp(stride)+1 )[1]) )
            output = input.new_empty(
                StrideConv2dFunction._output_size(ctx, input, weight, int_strides))
            print("THIS IS ERROR ",torch.exp(stride) )			
		
        print(' IntStrides ',int_strides,' Strides ', torch.exp(stride).detach().cpu().numpy().round(2),' Input ',input.shape,' Weight ',weight.shape, ' Output ',output.shape)		
        '''
		
		
        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]  # columns, ones
		
        #print(stride)		

        cur_im2col_step = min(ctx.im2col_step, input.size(0))		
        assert (input.size(0) % cur_im2col_step) == 0, 'im2col step (' +str(cur_im2col_step)+') must divide batchsize ('+str(input.size(0))+')'
        #print('cur_im2col_step',cur_im2col_step, input.shape)
        stride_conv_cuda.stride_conv_forward(
            input,
            weight,
            stride,
			bias,
            output,
            ctx.bufs_[0],
            ctx.bufs_[1],
            weight.size(3),
            weight.size(2),
            ctx.padding[1],
            ctx.padding[0],
            ctx.dilation[1],
            ctx.dilation[0],
            ctx.groups,
            cur_im2col_step)
        		
		
        return output
        
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, weight, strides, bias = ctx.saved_tensors

        grad_input = grad_stride = grad_weight = grad_bias =None

        cur_im2col_step = min(ctx.im2col_step, input.size(0))
        assert (input.size(0) %
                cur_im2col_step) == 0, 'im2col step must divide batchsize'

        grad_output = grad_output.contiguous()

        #print(strides)
        #print(input.shape)	
        #print(weight.shape)		
        grad_input = torch.zeros_like(input)
        grad_stride = torch.zeros_like(strides)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
		
        #print(ctx.needs_input_grad	, weight.shape	)
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[1]:
            grad_input = torch.zeros_like(input)
            grad_stride = torch.zeros_like(strides)
            #grad_stride = torch.zeros(1,1,2).to(strides.get_device())
            grad_weight = torch.zeros_like(weight)
			
            stride_conv_cuda.stride_conv_backward_input(
                input,
                strides,
                grad_output,
                grad_input,
                grad_stride,
                grad_bias,				
                weight,
                ctx.bufs_[0],
                weight.size(3),
                weight.size(2),
                ctx.padding[1],
                ctx.padding[0],
                ctx.dilation[1],
                ctx.dilation[0],
                ctx.groups,
                cur_im2col_step)
            #print("Grad after call ",grad_stride)
            #print("Grad after call ",grad_input)
            #grad_stride = torch.round(grad_stride*100)/100
        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros_like(weight)
            stride_conv_cuda.stride_conv_backward_parameters(
                input,
                strides,
                grad_output,
                grad_weight,
                ctx.bufs_[0],
                ctx.bufs_[1],
                weight.size(3),
                weight.size(2),
                ctx.padding[1],
                ctx.padding[0],
                ctx.dilation[1],
                ctx.dilation[0],
                ctx.groups,
                1,
                cur_im2col_step)
        #print(grad_weight)   
        #print(grad_bias)		
        return grad_input,  grad_weight, grad_stride, grad_bias, \
            None, None, None, None, None, None, None

    @staticmethod
    def _output_size(ctx, input, weight, int_strides):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = ctx.padding[d]
            kernel = ctx.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = int_strides[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                'convolution input is too small (output would be ' +
                'x'.join(map(str, output_size)) + ')')
        return output_size

class StrideAsInputConv2dFunction(Function):

    @staticmethod
    def forward(ctx,
                input,
                weight,
                stride_h,
                stride_w,				
				bias,
                padding=0,
                dilation=1,
                groups=1,
                im2col_step=32):
		
        if input is not None and input.dim() != 4:
            raise ValueError(
                f'Expected 4D tensor as input, got {input.dim()}D tensor \
                  instead.')
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.im2col_step = im2col_step
        ctx.stride_h =  stride_h
        ctx.stride_w =  stride_w
		
        input = input.type_as(weight)
        weight = weight.type_as(input)
		
		
        ctx.save_for_backward(input, weight, bias)
        
        int_strides = ( int(np.floor(stride_h)),int(np.floor(stride_w)) )
        output = input.new_empty(StrideConv2dFunction._output_size(ctx, input, weight, int_strides))
        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]  # columns, ones
		

        cur_im2col_step = min(ctx.im2col_step, input.size(0))		
        assert (input.size(0) % cur_im2col_step) == 0, 'im2col step (' +str(cur_im2col_step)+') must divide batchsize ('+str(input.size(0))+')'
        stride_input_conv_cuda_deterministic.stride_conv_forward(
            input,
            weight,
 			bias,
            output,
            ctx.bufs_[0],
            ctx.bufs_[1],
            weight.size(3),
            weight.size(2),
			stride_h,
			stride_w,
            ctx.padding[1],
            ctx.padding[0],
            ctx.dilation[1],
            ctx.dilation[0],
            ctx.groups,
            cur_im2col_step)
        		
		
        return output
        
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, weight,  bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias =None

        cur_im2col_step = min(ctx.im2col_step, input.size(0))
        assert (input.size(0) %
                cur_im2col_step) == 0, 'im2col step must divide batchsize'

        grad_output = grad_output.contiguous()

        #print(strides)
        #print(input.shape)	
        #print(weight.shape)		
        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias)
		
        #print(ctx.needs_input_grad	, weight.shape	)
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[1]:
            grad_input = torch.zeros_like(input)
            grad_weight = torch.zeros_like(weight)
			
            stride_input_conv_cuda_deterministic.stride_conv_backward_input(
                input,
                grad_output,
                grad_input,
                grad_bias,				
                weight,
                ctx.bufs_[0],
                weight.size(3),
                weight.size(2),
				ctx.stride_h,
			    ctx.stride_w,	
                ctx.padding[1],
                ctx.padding[0],
                ctx.dilation[1],
                ctx.dilation[0],
                ctx.groups,
                cur_im2col_step)

        
        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros_like(weight)
            stride_input_conv_cuda_deterministic.stride_conv_backward_parameters(
                input,
                grad_output,
                grad_weight,
                ctx.bufs_[0],
                ctx.bufs_[1],
                weight.size(3),
                weight.size(2),
				ctx.stride_h,
			    ctx.stride_w,				
                ctx.padding[1],
                ctx.padding[0],
                ctx.dilation[1],
                ctx.dilation[0],
                ctx.groups,
                1,
                cur_im2col_step)
        #print(grad_weight)    
        #print(grad_bias)		
        return grad_input,  grad_weight,  None, None,None, None, None, None

    @staticmethod
    def _output_size(ctx, input, weight, int_strides):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = ctx.padding[d]
            kernel = ctx.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = int_strides[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                'convolution input is too small (output would be ' +
                'x'.join(map(str, output_size)) + ')')
        return output_size
	
stride_conv2d = StrideConv2dFunction.apply	
stride_conv2d_as_input = StrideAsInputConv2dFunction.apply		
