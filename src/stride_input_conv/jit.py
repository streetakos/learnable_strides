from torch.utils.cpp_extension import load
stride_cuda = load(
    'stride_cuda', ['deform_conv.cpp', 'deform_conv_cuda.cu', 'deform_conv_cuda_kernel.cu'], verbose=True)
help(stride_cuda)

