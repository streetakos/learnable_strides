// Copyright (c) OpenMMLab. All rights reserved
#include "stride_conv_cuda_kernel.cu"
#include "pytorch_cuda_helper.hpp"
#include <cstdio>
#include <cmath>

void stride_im2col(Tensor data_im,  const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const float stride_h, const float stride_w ,const int pad_h, const int pad_w,
                       const int dilation_h, const int dilation_w,
                       const int parallel_imgs,
                       Tensor data_col) {
  // num_axes should be smaller than block size
  // todo: check parallel_imgs is correctly passed in


  int dH = (int)stride_h;
  int dW = (int)stride_w;

  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / dH + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / dW + 1;

  int num_kernels = channels * height_col * width_col * parallel_imgs;


  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.scalar_type(), "stride_im2col_gpu", ([&] {
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        scalar_t *data_col_ = data_col.data_ptr<scalar_t>();

        stride_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels),
                                       THREADS_PER_BLOCK, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_im_, height, width, ksize_h,  ksize_w,
            stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
            parallel_imgs, channels,
            height_col, width_col, data_col_);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

void stride_col2im_pytorch(Tensor data_col, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w,const float stride_h, const float stride_w, const int pad_h, const int pad_w,
                       const int dilation_h, const int dilation_w,
                       const int parallel_imgs, Tensor grad_im) {

   int dH = (int)stride_h;
   int dW = (int)stride_w;

  int num_kernels = channels * height * width;
  //Compute output height and width
  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / dH + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / dW + 1;
  //int num_kernels = channels * ksize_h * ksize_w * height_col * width_col * parallel_imgs;
  //int channel_per_deformable_group = channels / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "stride_col2im_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        scalar_t *grad_im_ = grad_im.data_ptr<scalar_t>();

        stride_col2im_pytorch_gpu_kernel<<<GET_BLOCKS(num_kernels),
                                       THREADS_PER_BLOCK, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_col_, channels, height, width,
            ksize_h, ksize_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
            dilation_w, parallel_imgs, height_col, width_col, grad_im_);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}

void stride_col2im(Tensor data_col, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w,const float stride_h, const float stride_w, const int pad_h, const int pad_w,
                       const int dilation_h, const int dilation_w,
                       const int parallel_imgs, Tensor grad_im) {

  int dH = (int)stride_h;
  int dW = (int)stride_w;

  // todo: make sure parallel_imgs is passed in correctly
  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / dH + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / dW + 1;
  int num_kernels = channels * ksize_h * ksize_w * height_col * width_col * parallel_imgs;
  //int channel_per_deformable_group = channels / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "stride_col2im_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        scalar_t *grad_im_ = grad_im.data_ptr<scalar_t>();

        stride_col2im_gpu_kernel<<<GET_BLOCKS(num_kernels),
                                       THREADS_PER_BLOCK, 0,
                                       at::cuda::getCurrentCUDAStream()>>>(
            num_kernels, data_col_, channels, height, width,
            ksize_h, ksize_w, stride_h,stride_w,pad_h, pad_w, dilation_h,
            dilation_w, parallel_imgs, height_col, width_col, grad_im_);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}


void stride_conv_shape_check(Tensor input, Tensor *gradOutput,
                             Tensor weight, int kH, int kW,float stride_h, float stride_w,
                             int padH, int padW, int dilationH, int dilationW,
                             int group) {

  TORCH_CHECK(
      weight.ndimension() == 4,
      "4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, but got: %s",
      weight.ndimension());

  TORCH_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");

  TORCH_CHECK(kW > 0 && kH > 0,
              "kernel size should be greater than zero, but got kH: %d kW: %d",
              kH, kW);

  TORCH_CHECK((weight.size(2) == kH && weight.size(3) == kW),
              "kernel size should be consistent with weight, ",
              "but got kH: %d kW: %d weight.size(2): %d, weight.size(3): %d",
              kH, kW, weight.size(2), weight.size(3));

  float dfH = stride_h;
  float dfW = stride_w;

  TORCH_CHECK( dfH >= 1.0 && dfW >= 1.0,
              "stride should be greater equal than one, but got dH: %f dW: %f", dfH,
              dfW);

  TORCH_CHECK(
      dilationW > 0 && dilationH > 0,
      "dilation should be greater than 0, but got dilationH: %d dilationW: %d",
      dilationH, dilationW);

  int ndim = input.ndimension();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  TORCH_CHECK(ndim == 3 || ndim == 4,
              "3D or 4D input tensor expected but got: %s", ndim);

  int dH = (int)stride_h;
  int dW = (int)stride_w;

  long nInputPlane = weight.size(1) * group;
  long inputHeight = input.size(dimh);
  long inputWidth = input.size(dimw);
  long nOutputPlane = weight.size(0);
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;




  if (outputWidth < 1 || outputHeight < 1)
    AT_ERROR(
        "Given input size: (%ld x %ld x %ld). "
        "Calculated output size: (%ld x %ld x %ld). Output size is too small",
        nInputPlane, inputHeight, inputWidth, nOutputPlane, outputHeight,
        outputWidth);

  TORCH_CHECK(input.size(1) == nInputPlane,
              "invalid number of input planes, expected: %d, but got: %d",
              nInputPlane, input.size(1));

  TORCH_CHECK((inputHeight >= kH && inputWidth >= kW),
              "input image is smaller than kernel");



  if (gradOutput != NULL) {
    TORCH_CHECK(
        gradOutput->size(dimf) == nOutputPlane,
        "invalid number of gradOutput planes, expected: %d, but got: %d",
        nOutputPlane, gradOutput->size(dimf));

    TORCH_CHECK(
        (gradOutput->size(dimh) == outputHeight &&
         gradOutput->size(dimw) == outputWidth),
        "invalid size of gradOutput, expected height: %d width: %d , but "
        "got height: %d width: %d",
        outputHeight, outputWidth, gradOutput->size(dimh),
        gradOutput->size(dimw));
  }
}

void StrideConvForwardCUDAKernelLauncher(Tensor input, Tensor weight,
                                         Tensor bias, Tensor output,
                                         Tensor columns, Tensor ones, int kW,
                                         int kH, float stride_h, float stride_w,
                                         int padW, int padH, int dilationW, int dilationH,
                                         int group, int im2col_step) {


  at::DeviceGuard guard(input.device());


  stride_conv_shape_check(input, NULL, weight, kH, kW,stride_h,stride_w,padH,
                          padW, dilationH, dilationW, group );

  int batch = 1;
  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input.unsqueeze_(0);
  }


  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);

  // Get the integer part of strides in order to compute the output
  int dH = (int)stride_h;
  int dW = (int)stride_w;

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;


  output = output.view({batchSize / im2col_step, im2col_step, nOutputPlane,
                        outputHeight, outputWidth});

  columns = at::zeros(
      {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
      input.options());

  if (ones.ndimension() != 2 ||
      ones.size(0) * ones.size(1) < outputHeight * outputWidth) {
    ones = at::ones({outputHeight, outputWidth}, input.options());
  }

  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane,
                      inputHeight, inputWidth});


  Tensor output_buffer = at::zeros({batchSize / im2col_step, nOutputPlane,
                                    im2col_step * outputHeight, outputWidth},
                                   output.options());

  output_buffer = output_buffer.view(
      {output_buffer.size(0), group, output_buffer.size(1) / group,
       output_buffer.size(2), output_buffer.size(3)});


  Tensor bias_g = bias.view({group, nOutputPlane/group});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    stride_im2col(input[elt], nInputPlane, inputHeight,
                      inputWidth, kH, kW,   stride_h,  stride_w, padH, padW, dilationH,
                      dilationW, im2col_step, columns);

    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    weight = weight.view({group, weight.size(0) / group, weight.size(1),
                          weight.size(2), weight.size(3)});

    for (int g = 0; g < group; g++) {

	  output_buffer[elt][g]  = (output_buffer[elt][g]
                                  .flatten(1).addmm_(weight[g].flatten(1), columns[g]) + bias_g[g].view({bias_g[g].size(0),1})  ).view_as(output_buffer[elt][g]) ;
	  }
    columns = columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
                          weight.size(3), weight.size(4)});
  }

  output_buffer = output_buffer.view(
      {output_buffer.size(0), output_buffer.size(1) * output_buffer.size(2),
       output_buffer.size(3), output_buffer.size(4)});

  output_buffer = output_buffer.view({batchSize / im2col_step, nOutputPlane,
                                      im2col_step, outputHeight, outputWidth});
  output_buffer.transpose_(1, 2);
  output.copy_(output_buffer);
  output = output.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});

  if (batch == 0) {
    output = output.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
  }

}

void StrideConvBackwardInputCUDAKernelLauncher(
    Tensor input, Tensor gradOutput, Tensor gradInput,
    Tensor gradBias, Tensor weight, Tensor columns, int kW, int kH, float stride_h, float stride_w,
	int padW, int padH, int dilationW, int dilationH, int group,
    int im2col_step) {

  at::DeviceGuard guard(input.device());


  int batch = 1;

  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.view({1, input.size(0), input.size(1), input.size(2)});
    gradOutput = gradOutput.view({1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);

  long ws0 = 	weight.size(0);
  long ws1 = 	weight.size(1);
  long ws2 = 	weight.size(2);
  long ws3 = 	weight.size(3);

  int dH = (int)stride_h;
  int dW = (int)stride_w;

  long outputWidth = (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  //TORCH_CHECK((offset.size(0) == batchSize), 3, "invalid batch size of offset");
  gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
  columns = at::zeros({nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
      input.options());

  // change order of grad output
  gradOutput = gradOutput.view({batchSize / im2col_step, im2col_step,  nOutputPlane, outputHeight, outputWidth});
  gradOutput.transpose_(1, 2);
  gradInput = gradInput.view({batchSize / im2col_step, im2col_step, nInputPlane, inputHeight, inputWidth});
  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane, inputHeight, inputWidth});

  gradBias = gradBias.view({group, nOutputPlane/group});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
	// Multiply each weight with each outputpixel and sum the output dimentions via multiplication
    // divide into groups
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    weight = weight.view({ group, weight.size(0) / group, weight.size(1), weight.size(2), weight.size(3)});
    gradOutput = gradOutput.view({ gradOutput.size(0), group, gradOutput.size(1) / group, gradOutput.size(2), gradOutput.size(3), gradOutput.size(4)});

    // From the chain rule:
	   //G_offset = gradOutput * W * G_interpolated_value,  Compute: gradOutput * W
    for (int g = 0; g < group; g++) {
      columns[g] = columns[g].addmm_(weight[g].flatten(1).transpose(0, 1),
                                     gradOutput[elt][g].flatten(1), 0.0f, 1.0f);
      gradBias[g] = gradOutput[elt][g].transpose(0, 1).sum( {0,2,3}, false) ; ///////////////////////////////////////////////////////////////////
    }

    columns = columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    gradOutput = gradOutput.view({gradOutput.size(0), gradOutput.size(1) * gradOutput.size(2), gradOutput.size(3), gradOutput.size(4), gradOutput.size(5)});

    weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
                          weight.size(3), weight.size(4)});

  }



  gradOutput.transpose_(1, 2);
  gradOutput = gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Code adopted from pytorch for avoiding atomicadd operation (Deterministic)
  ///*
  gradInput = gradInput.view({batchSize , nInputPlane, inputHeight, inputWidth});
  weight = weight.view({  ws0 , ws1, ws2, ws3});

  for (int elt = 0; elt < batchSize ; elt++) {
	Tensor columnsa = at::zeros({nInputPlane * kW * kH, outputHeight * outputWidth},input.options());

	columnsa = columnsa.addmm_(weight.flatten(1).transpose(0, 1),
                                     gradOutput[elt].flatten(1), 0.0f, 1.0f);

	stride_col2im_pytorch(columnsa, nInputPlane, inputHeight,
                      inputWidth, kH, kW, stride_h, stride_w, padH, padW, dilationH,
                      dilationW, im2col_step, gradInput[elt]);
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////


  gradInput = gradInput.view({batchSize, nInputPlane, inputHeight, inputWidth});
  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  gradBias = gradBias.view({nOutputPlane});


  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
    gradInput = gradInput.view({nInputPlane, inputHeight, inputWidth});
  }

}

void StrideConvBackwardParametersCUDAKernelLauncher(
    Tensor input, Tensor gradOutput, Tensor gradWeight,
    Tensor columns, Tensor ones, int kW, int kH, float stride_h, float stride_w,
    int padW, int padH, int dilationW, int dilationH, int group,
    float scale, int im2col_step) {

  at::DeviceGuard guard(input.device());

  int batch = 1;

  if (input.ndimension() == 3) {
    // Force batch
    batch = 0;
    input = input.view( at::IntList({1, input.size(0), input.size(1), input.size(2)}));
    gradOutput = gradOutput.view({1, gradOutput.size(0), gradOutput.size(1), gradOutput.size(2)});
  }

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = gradWeight.size(0);

  int dH = (int)stride_h;
  int dW = (int)stride_w;

  long outputWidth =  (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight = (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  //TORCH_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");

  columns = at::zeros({nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth}, input.options());

  gradOutput = gradOutput.view({batchSize / im2col_step, im2col_step, nOutputPlane, outputHeight, outputWidth});
  gradOutput.transpose_(1, 2);

  Tensor gradOutputBuffer = at::zeros_like(gradOutput);
  gradOutputBuffer = gradOutputBuffer.view({batchSize / im2col_step, nOutputPlane, im2col_step, outputHeight, outputWidth});
  gradOutputBuffer = gradOutputBuffer.contiguous();
  gradOutputBuffer.copy_(gradOutput);
  gradOutputBuffer = gradOutputBuffer.view({batchSize / im2col_step, nOutputPlane, im2col_step * outputHeight, outputWidth});

  gradOutput.transpose_(1, 2);
  gradOutput =  gradOutput.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane, inputHeight, inputWidth});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    stride_im2col(input[elt], nInputPlane, inputHeight,
                      inputWidth, kH, kW, stride_h, stride_w, padH, padW, dilationH,
                      dilationW, im2col_step, columns);

    // divide into group
    gradOutputBuffer = gradOutputBuffer.view( {gradOutputBuffer.size(0), group, gradOutputBuffer.size(1) / group, gradOutputBuffer.size(2), gradOutputBuffer.size(3)});
    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    gradWeight = gradWeight.view({group, gradWeight.size(0) / group, gradWeight.size(1), gradWeight.size(2), gradWeight.size(3)});

    for (int g = 0; g < group; g++) {
      gradWeight[g] = gradWeight[g].flatten(1).addmm_(gradOutputBuffer[elt][g].flatten(1),
                                                     columns[g].transpose(1, 0), 1.0, scale).view_as(gradWeight[g]);
    }
    gradOutputBuffer = gradOutputBuffer.view( {gradOutputBuffer.size(0), gradOutputBuffer.size(1) * gradOutputBuffer.size(2), gradOutputBuffer.size(3), gradOutputBuffer.size(4)});
    columns = columns.view({columns.size(0) * columns.size(1), columns.size(2)});
    gradWeight = gradWeight.view({gradWeight.size(0) * gradWeight.size(1), gradWeight.size(2), gradWeight.size(3), gradWeight.size(4)});
  }

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  //offset = offset.view( {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  if (batch == 0) {
    gradOutput = gradOutput.view({nOutputPlane, outputHeight, outputWidth});
    input = input.view({nInputPlane, inputHeight, inputWidth});
  }

}
