/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer
 *****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer
 *********************
 *
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file modulated_deformable_im2col.cuh
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, dilation, and offset.
 * These functions are mainly used in deformable convolution operators.
 * \ref: https://arxiv.org/abs/1703.06211
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu, Dazhi Cheng
 */

// modified from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu

#ifndef DEFORM_CONV_CUDA_KERNEL_CUH
#define DEFORM_CONV_CUDA_KERNEL_CUH

#include <float.h>
#ifdef MMCV_WITH_TRT
#include "common_cuda_helper.hpp"
#else  // MMCV_WITH_TRT
#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else  // MMCV_USE_PARROTS
#include "pytorch_cuda_helper.hpp"
#endif  // MMCV_USE_PARROTS
#endif  // MMCV_WITH_TRT

template <typename T>
__device__ T deformable_im2col_bilinear(const T *input, const int data_width,
                                        const int height, const int width, T h,
                                        T w) {
  if (h <= -1 || height <= h || w <= -1 || width <= w) {
    return 0;
  }

  int h_low = floorf(h);
  int w_low = floorf(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh, hw = 1 - lw;

  T v1 = 0;
  if (h_low >= 0 && w_low >= 0) v1 = input[h_low * data_width + w_low];
  T v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = input[h_low * data_width + w_high];
  T v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = input[h_high * data_width + w_low];
  T v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = input[h_high * data_width + w_high];

  T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename T>
__device__ T get_gradient_weight(T argmax_h, T argmax_w, const int h,
                                 const int w, const int height,
                                 const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floorf(argmax_h);
  int argmax_w_low = floorf(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

template <typename T>
__device__ T get_coordinate_weight(T argmax_h, T argmax_w, const int height,
                                   const int width, const T *im_data,
                                   const int data_width, const int bp_dir) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floorf(argmax_h);
  int argmax_w_low = floorf(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) *
                im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) *
                im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) *
                im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) *
                im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) *
                im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) *
                im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) *
                im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) *
                im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template <typename T>
__global__ void stride_im2col_gpu_kernel(
    const int n, const T *data_im, const int height,
    const int width, const int kernel_h, const int kernel_w, const float stride_h, const float stride_w,
    const int pad_h,   const int pad_w,
    const int dilation_h, const int dilation_w,
    const int batch_size,
    const int num_channels, const int height_col,
    const int width_col, T *data_col) {

  // For each GPU thread = index
  CUDA_1D_KERNEL_LOOP(index, n) {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;


    T *data_col_ptr =
        data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;

	  const T *data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;


    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {

		    T val = static_cast<T>(0);
		    const T h_im = h_col * stride_h - pad_h + i * dilation_h;
        const T w_im = w_col * stride_w - pad_w + j * dilation_w;

		    // If in bounds
		    if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
          val = deformable_im2col_bilinear(data_im_ptr, width, height, width,
                                           h_im, w_im);
        *data_col_ptr = val;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}



template <typename T>
__global__ void  stride_col2im_pytorch_gpu_kernel(
    const int n,   const T  *data_col,  const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w,
	  const float stride_h, const float stride_w,
	  const int pad_height,  const int pad_width,
    const int dilation_height,  const int dilation_width, const int batch_size,  const int height_col,
    const int width_col,  T* data_im) {


  CUDA_1D_KERNEL_LOOP(index, n) {
	  const int stride_width = (int)stride_w;
    const int stride_height = (int)stride_h;
    T val = 0;
    const int w_im = index % width + pad_width;
    const int h_im = (index / width) % height + pad_height;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_width + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_height + 1;

    // compute the start and end of the output
    const int w_col_start = (w_im < kernel_extent_w)
        ? 0
        : (w_im - kernel_extent_w) / stride_width + 1;
    const int w_col_end = ::min(w_im / stride_width + 1, width_col);
    const int h_col_start = (h_im < kernel_extent_h)
        ? 0
        : (h_im - kernel_extent_h) / stride_height + 1;
    const int h_col_end = ::min(h_im / stride_height + 1, height_col);

    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {

        int h_k = (h_im - h_col * stride_height);
        int w_k = (w_im - w_col * stride_width);
        if (h_k % dilation_height == 0 && w_k % dilation_width == 0) {
          h_k /= dilation_height;
          w_k /= dilation_width;

          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col + h_col) * width_col + w_col;

		  T h_k_m = (h_im - h_col * stride_h);
          T w_k_m = (w_im - w_col * stride_w);
		  T weight = get_gradient_weight(h_k_m, w_k_m, h_k, w_k, height, width)	;
          val += weight* data_col[data_col_index];
        }
      }
    }
    data_im[index] = val;
  }
}

template <typename T>
__global__ void stride_col2im_gpu_kernel(
    const int n, const T *data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const float stride_h, const float stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w,
    const int batch_size, const int height_col, const int width_col,
    T *grad_im) {


  CUDA_1D_KERNEL_LOOP(index, n) {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i = (index / width_col / height_col / batch_size) / kernel_w % kernel_h;
    const int c =  index / width_col / height_col / batch_size / kernel_w / kernel_h;


    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;


    const T cur_inv_h_data = h_out * stride_h - pad_h + i * dilation_h;// + offset_h;
    const T cur_inv_w_data = w_out * stride_w - pad_w + j * dilation_w;// + offset_w;

    const T cur_top_grad = data_col[index];
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;

	//T val =  0;
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {

        if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 &&
            cur_w + dx < width && abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1) {

          int cur_bottom_grad_pos = ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          T weight = get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);

          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);

        }
      }

    }
  }

}

#endif  // DEFORM_CONV_CUDA_KERNEL_CUH
