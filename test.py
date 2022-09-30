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
        c_in = 4
        c_out = 4
        batch_size = 1
        ks = 2	# Not to be changed
        pad = 0		
        stride = 1
        use_bias = False		
        bi = torch.randn(c_out)		
        grad_out =  torch.randn_like(out)		
		
        # Define input		
        repeated_input = np.repeat(input, batch_size, axis=0)
        repeated_input = np.repeat(input, c_in, axis=1)
        x = torch.tensor(repeated_input, device=device, dtype=dtype)
        x.requires_grad = True

        # Define weights
        deform_weight = [[[0.4, 0.2, 0.1, 0.9]]]		
        deform_weight = np.reshape(deform_weight,[1,1,ks,ks])
        deform_weight = np.repeat(deform_weight, c_in, axis=0)
        deform_weight = np.repeat(deform_weight, c_out, axis=1)
        w_ = torch.nn.Parameter(torch.Tensor(deform_weight).reshape(c_out, c_in, ks, ks))		

        # Define strided conv		
        model = StrideConv2d(in_channels=c_in, out_channels=c_out, kernel_size=2, padding=pad, bias=use_bias)
        model.weight.data = w_#torch.nn.Parameter(torch.Tensor(deform_weight).reshape(1, 1, 2, 2))
        # Strides initialized wih ones		
        model.stride.data = model.stride.data + (stride-1.0)

        if use_bias:		
            model.bias = torch.nn.Parameter(bi)
        if device == 'cuda':
            model.cuda()
        model.type(dtype)
        
        out = model(x) 
		
        out.backward(grad_out)	#ones_like	
		
        # Define a standard convolutional model
        # All operations must yield the same results as the StrideConv with integer strides
        x_2 = torch.tensor(repeated_input, device=device, dtype=dtype)
        x_2.requires_grad = True	
        model_2 = torch.nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=2, padding=pad, stride=stride, bias=use_bias)
        if use_bias:		
            model_2.bias = torch.nn.Parameter(bi)        
        model_2.weight.data = w_#torch.nn.Parameter(torch.Tensor(deform_weight).reshape(1, 1, 2, 2))
        if device == 'cuda':
            model_2.cuda()
        model_2.type(dtype)
        out_2 = model_2(x_2) 
        out_2.backward(grad_out)		
        #print(out)		
		
        print(out)		
        print(out_2)		 
        print(model)		
        print(model_2)		
	
		
        # Check if the gradients and the output are all close		
        assert np.allclose(model_2.weight.grad.detach().cpu().numpy()
						  ,model.weight.grad.detach().cpu().numpy(), threshold)
		
        if use_bias:		
            assert np.allclose(model_2.bias.grad.detach().cpu().numpy()
						  ,model.bias.grad.detach().cpu().numpy(), threshold)		
        assert np.allclose(out.data.detach().cpu().numpy(),
						   out_2.data.detach().cpu().numpy(),threshold)
        assert np.allclose(x_2.grad.detach().cpu().numpy()
						  ,x.grad.detach().cpu().numpy(), threshold)
        		
        '''		
        out.backward(torch.ones_like(out))

        assert np.allclose(out.data.detach().cpu().numpy(), repeated_gt_out,
                           threshold)
        assert np.allclose(x.grad.detach().cpu().numpy(), repeated_gt_x_grad,
                           threshold)
        # the batch size of the input is increased which results in
        # a larger gradient so we need to divide by the batch_size
        assert np.allclose(
            model.conv_offset.weight.grad.detach().cpu().numpy() / batch_size,
            gt_offset_weight_grad, threshold)
        assert np.allclose(
            model.conv_offset.bias.grad.detach().cpu().numpy() / batch_size,
            gt_offset_bias_grad, threshold)
        assert np.allclose(
            model.weight.grad.detach().cpu().numpy() / batch_size,
            gt_deform_weight_grad, threshold)
        
        from deform_conv import DeformConv2d

        # test bias
        model = DeformConv2d(1, 1, 2, stride=1, padding=0)
        assert not hasattr(model, 'bias')
        # test bias=True
        with pytest.raises(AssertionError):
            model = DeformConv2d(1, 1, 2, stride=1, padding=0, bias=True)
        # test in_channels % group != 0
        with pytest.raises(AssertionError):
            model = DeformConv2d(3, 2, 3, groups=2)
        # test out_channels % group != 0
        with pytest.raises(AssertionError):
            model = DeformConv2d(3, 4, 3, groups=3)
        '''        
    '''
    def test_deformconv(self):
        self._test_deformconv(torch.double, device='cpu')
        self._test_deformconv(torch.float, device='cpu', threshold=1e-1)
        self._test_deformconv(torch.double)
        self._test_deformconv(torch.float)
        self._test_deformconv(torch.half, threshold=1e-1)
        # test batch_size < im2col_step
        self._test_deformconv(torch.float, batch_size=1, im2col_step=2)
        # test bach_size % im2col_step != 0
        with pytest.raises(
                AssertionError,
                match='batch size must be divisible by im2col_step'):
            self._test_deformconv(torch.float, batch_size=10, im2col_step=3)
    '''

				
td = TestStrideConv()
td._test_strideconv()



