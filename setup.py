from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="rad_polygon_mask",
    packages=['rad_polygon_mask'],
    ext_modules=[
        CUDAExtension(
            name="rad_polygon_mask._C",
            sources=[
            # "cuda_rasterizer/rasterizer_impl.cu",
            # "cuda_rasterizer/forward.cu",
            "polymask_impl/polymask_forward.cu",
            "polygon_mask.cu",
            "ext.cpp"],)
            # extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
