// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include <cstdio>

#ifdef MMCV_WITH_CUDA
void StrideConvForwardCUDAKernelLauncher(Tensor input, Tensor weight,
                                         Tensor strides, Tensor bias,  Tensor output,
                                         Tensor columns, Tensor ones, int kW,
                                         int kH, int padW,
                                         int padH, int dilationW, int dilationH,
                                         int group, 
                                         int im2col_step);

void StrideConvBackwardInputCUDAKernelLauncher(
    Tensor input, Tensor strides, Tensor gradOutput, Tensor gradInput,
    Tensor gradStrides, Tensor gradBias, Tensor weight, Tensor columns, int kW, int kH, 
	int padW, int padH, int dilationW, int dilationH, int group,
	int im2col_step);

void StrideConvBackwardParametersCUDAKernelLauncher(
    Tensor input, Tensor strides, Tensor gradOutput, Tensor gradWeight,
    Tensor columns, Tensor ones, int kW, int kH, int padW,
    int padH, int dilationW, int dilationH, int group,
    float scale, int im2col_step);

void stride_conv_forward_cuda(Tensor input, Tensor weight, Tensor strides,  Tensor bias,
                              Tensor output, Tensor columns, Tensor ones,
                              int kW, int kH, int padW,
                              int padH, int dilationW, int dilationH, int group, int im2col_step) {
  StrideConvForwardCUDAKernelLauncher(
      input, weight, strides, bias,  output, columns, ones, kW, kH, padW, padH,
      dilationW, dilationH, group, im2col_step);
}

void stride_conv_backward_input_cuda(Tensor input, Tensor strides,
                                     Tensor gradOutput, Tensor gradInput,
                                     Tensor gradStrides,Tensor gradBias, Tensor weight,
                                     Tensor columns, int kW, int kH, int padW, int padH, int dilationW,
                                     int dilationH, int group,
                                     int im2col_step) {
  StrideConvBackwardInputCUDAKernelLauncher(
      input, strides, gradOutput, gradInput, gradStrides, gradBias, weight, columns, kW, kH,
      padW, padH, dilationW, dilationH, group, 
      im2col_step);
}

void stride_conv_backward_parameters_cuda(
    Tensor input, Tensor strides, Tensor gradOutput, Tensor gradWeight,
    Tensor columns, Tensor ones, int kW, int kH, int padW,
    int padH, int dilationW, int dilationH, int group,
    float scale, int im2col_step) {
  StrideConvBackwardParametersCUDAKernelLauncher(
      input, strides, gradOutput, gradWeight, columns, ones, kW, kH,
      padW, padH, dilationW, dilationH, group, scale,
      im2col_step);
}
#endif

void stride_conv_forward(Tensor input, Tensor weight, Tensor strides, Tensor bias, 
                         Tensor output, Tensor columns, Tensor ones, int kW,
                         int kH, int padW, int padH,
                         int dilationW, int dilationH, int group, int im2col_step) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(strides);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(output);
    CHECK_CUDA_INPUT(columns);
    CHECK_CUDA_INPUT(ones);
    CHECK_CUDA_INPUT(bias);
	//printf("Call to Cpp files completed \n");
	  
    stride_conv_forward_cuda(input, weight, strides, bias,  output, columns, ones, kW,
                             kH, padW, padH, dilationW, dilationH,
                             group, im2col_step);
#else
    AT_ERROR("DeformConv is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("DeformConv is not implemented on CPU");
  }
}

void stride_conv_backward_input(Tensor input, Tensor strides, Tensor gradOutput,
                                Tensor gradInput, Tensor gradStrides, Tensor gradBias,
                                Tensor weight, Tensor columns, int kW, int kH, int padW, int padH,
                                int dilationW, int dilationH, int group, int im2col_step) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(strides);
    CHECK_CUDA_INPUT(gradOutput);
    CHECK_CUDA_INPUT(gradInput);
    CHECK_CUDA_INPUT(gradStrides);
    CHECK_CUDA_INPUT(weight);
    CHECK_CUDA_INPUT(columns);
	CHECK_CUDA_INPUT(gradBias);  

    stride_conv_backward_input_cuda(input, strides, gradOutput, gradInput,
                                    gradStrides,gradBias, weight, columns, kW, kH,
                                    padW, padH, dilationW, dilationH, group,
                                    im2col_step);
#else
    AT_ERROR("DeformConv is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("DeformConv is not implemented on CPU");
  }
}

void stride_conv_backward_parameters(Tensor input, Tensor strides,
                                     Tensor gradOutput, Tensor gradWeight,
                                     Tensor columns, Tensor ones, int kW,
                                     int kH, int padW, int padH,
                                     int dilationW, int dilationH, int group,
                                     float scale,
                                     int im2col_step) {
  if (input.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(input);
    CHECK_CUDA_INPUT(strides);
    CHECK_CUDA_INPUT(gradOutput);
    CHECK_CUDA_INPUT(gradWeight);
    CHECK_CUDA_INPUT(columns);
    CHECK_CUDA_INPUT(ones);

    stride_conv_backward_parameters_cuda(input, strides, gradOutput, gradWeight,
                                         columns, ones, kW, kH, padW,
                                         padH, dilationW, dilationH, group,
                                         scale, im2col_step);
#else
    AT_ERROR("DeformConv is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("DeformConv is not implemented on CPU");
  }
}






PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("stride_conv_backward_parameters", &stride_conv_backward_parameters, "stride_conv_backward_parameters (CUDA)");
  m.def("stride_conv_backward_input", &stride_conv_backward_input, "stride_conv_backward_input (CUDA)");
  m.def("stride_conv_forward", &stride_conv_forward, "stride_conv_forward (CUDA)");
}

