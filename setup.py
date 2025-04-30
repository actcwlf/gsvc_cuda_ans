from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os


proj_dir = os.path.dirname(os.path.abspath(__file__))


setup(
    name="gsvc_cuda_ans",
    packages=['gsvc_cuda_ans'],
    ext_modules=[
        CUDAExtension(
            name="gsvc_cuda_ans._C",
            sources=[
            "src/gaussian.cu",
            "src/simple_ans.cu",
            "binding.cpp"],
            extra_compile_args={
                "nvcc": [
                    "-I" + proj_dir + '/src'
                    
                    ],
                "cxx": [
                    "-I" + proj_dir + '/src'
                    
                    ],
                })
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
