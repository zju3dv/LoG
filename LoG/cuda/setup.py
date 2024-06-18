# import glob
# import os.path as osp
# from setuptools import setup
# from torch.utils.cpp_extension import CUDAExtension, BuildExtension


# ROOT_DIR = osp.dirname(osp.abspath(__file__))
# include_dirs = [osp.join(ROOT_DIR, "include")]

# sources = glob.glob('*.cpp')+glob.glob('*.cu')


# setup(
#     name='compute_radius',
#     version='1.0',
#     author='kwea123',
#     author_email='kwea123@gmail.com',
#     description='cppcuda_tutorial',
#     long_description='compute',
#     ext_modules=[
#         CUDAExtension(
#             name='compute',
#             sources=sources,
#             include_dirs=include_dirs,
#             extra_compile_args={'cxx': ['-O2'],
#                                 'nvcc': ['-O2']}
#         )
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     }
# )


from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="compute_radius",
    # packages=['compute_radius'],
    ext_modules=[
        CUDAExtension(
            name="compute",
            sources=[
            "compute_radius_kernel.cu",
            ],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)