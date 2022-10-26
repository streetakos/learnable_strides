from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

define_macros = []
define_macros += [('MMCV_WITH_CUDA', None)]
define_macros += [('MMCV_WITH_HIP', None)]


setup(
    name='stride_input_conv_cuda_deterministic',
    ext_modules=[
        CUDAExtension('stride_input_conv_cuda_deterministic', [
            'stride_conv.cpp',
            'stride_conv_cuda.cu',
            'stride_conv_cuda_kernel.cu',
        ],define_macros=define_macros),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },zip_safe=False)
