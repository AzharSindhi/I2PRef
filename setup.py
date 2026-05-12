import os.path as osp
from setuptools import find_packages, setup
import os
os.environ["TORCH_CUDA_ARCH_LIST"]="7.0;7.5;8.0;8.6"

requirements = ["hydra-core==0.11.3"]


exec(open(osp.join("pointnet2", "_version.py")).read())

setup(
    name="pointnet2",
    version=__version__,
    packages=find_packages(),
    install_requires=requirements,
)
