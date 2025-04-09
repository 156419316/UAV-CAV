from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointpillars',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='detector.pointpillars.ops.voxel_op',
            sources=[
                'detector/pointpillars/ops/voxelization/voxelization.cpp',
                'detector/pointpillars/ops/voxelization/voxelization_cpu.cpp',
                'detector/pointpillars/ops/voxelization/voxelization_cuda.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        ),
        CUDAExtension(
            name='detector.pointpillars.ops.iou3d_op',
            sources=[
                'detector/pointpillars/ops/iou3d/iou3d.cpp',
                'detector/pointpillars/ops/iou3d/iou3d_kernel.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False
)