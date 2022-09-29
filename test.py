import numpy as np
import pytest
import torch

#from mmcv.utils import TORCH_VERSION, digit_version

input = [[[[10., 2., 3.], [0., 1., 2.], [30., 5., 2.]]]]
offset_weight = [[[0.1, 0.4, 0.6, 0.1]], [[0.3, 0.2, 0.1, 0.3]],
                 [[0.5, 0.5, 0.2, 0.8]], [[0.8, 0.3, 0.9, 0.1]],
                 [[0.3, 0.1, 0.2, 0.5]], [[0.3, 0.7, 0.5, 0.3]],
                 [[0.6, 0.2, 0.5, 0.3]], [[0.4, 0.1, 0.8, 0.4]]]

offset_bias = [0.7, 0.1, 0.8, 0.5, 0.6, 0.5, 0.4, 0.7]

deform_weight = [[[0.4, 0.2, 0.1, 0.9]]]

gt_out = [[[[1.650, 0.], [0.000, 0.]]]]

gt_x_grad = [[[[-0.666, 0.204, 0.000], [0.030, -0.416, 0.012],
               [0.000, 0.252, 0.129]]]]

gt_offset_weight_grad = [[[[1.44, 2.88], [0.00, 1.44]]],
                         [[[-0.72, -1.44], [0.00, -0.72]]],
                         [[[0.00, 0.00], [0.00, 0.00]]],
                         [[[0.00, 0.00], [0.00, 0.00]]],
                         [[[-0.10, -0.20], [0.00, -0.10]]],
                         [[[-0.08, -0.16], [0.00, -0.08]]],
                         [[[-0.54, -1.08], [0.00, -0.54]]],
                         [[[-0.54, -1.08], [0.00, -0.54]]]]
gt_offset_bias_grad = [1.44, -0.72, 0., 0., -0.10, -0.08, -0.54, -0.54],
gt_deform_weight_grad = [[[[3.62, 0.], [0.40, 0.18]]]]


class TestStrideConv:

    def _test_strideconv(self,
                         dtype=torch.float,
                         threshold=1e-3,
                         device='cuda',
                         batch_size=10,
                         im2col_step=2):
		
        if not torch.cuda.is_available() and device == 'cuda':
            pytest.skip('test requires GPU')
			
        from src.strides import StrideConv2d
		
        # Hyperparameters
        c_in = 1
        c_out = 1
        batch_size = 1
        ks = 2	# Not to be changed
        pad = 0		
        stride = 1
		
        # define input		
        repeated_input = np.repeat(input, batch_size, axis=0)
        repeated_input = np.repeat(input, c_in, axis=1)
        x = torch.tensor(repeated_input, device=device, dtype=dtype)
        x.requires_grad = True

        #Define weights
        deform_weight = [[[0.4, 0.2, 0.1, 0.9]]]		
        deform_weight = np.reshape(deform_weight,[1,1,ks,ks])
        deform_weight = np.repeat(deform_weight, c_in, axis=0)
        deform_weight = np.repeat(deform_weight, c_out, axis=1)
        w_ = torch.nn.Parameter(torch.Tensor(deform_weight).reshape(c_out, c_in, ks, ks))		

        # Define strided conv		
        model = StrideConv2d(in_channels=c_in, out_channels=c_out, kernel_size=2, padding=pad)
        model.weight.data = w_#torch.nn.Parameter(torch.Tensor(deform_weight).reshape(1, 1, 2, 2))
        # Strides initialized wih ones		
        model.stride.data = model.stride.data + (stride-1.0)	
        if device == 'cuda':
            model.cuda()
        model.type(dtype)
        out = model(x) 
        out.backward(torch.ones_like(out))		
		
        # Define a standard convolutional model
        # All operations must yield the same results as the StrideConv with integer strides
        x_2 = torch.tensor(repeated_input, device=device, dtype=dtype)
        x_2.requires_grad = True	
        model_2 = torch.nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=2, padding=pad, stride=stride, bias=False)
        model_2.weight.data = w_#torch.nn.Parameter(torch.Tensor(deform_weight).reshape(1, 1, 2, 2))
        if device == 'cuda':
            model_2.cuda()
        model_2.type(dtype)
        out_2 = model_2(x_2) 
        out_2.backward(torch.ones_like(out_2))			

	
        # Check if the gradients and the output are all close		
        assert np.allclose(model_2.weight.grad.detach().cpu().numpy()
						  ,model.weight.grad.detach().cpu().numpy(), threshold)
        assert np.allclose(out.data.detach().cpu().numpy(),
						   out_2.data.detach().cpu().numpy(),threshold)
        assert np.allclose(x_2.grad.detach().cpu().numpy()
						  ,x.grad.detach().cpu().numpy(), threshold)
        		
				
td = TestStrideConv()
td._test_strideconv()



