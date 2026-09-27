from setuptools import find_namespace_packages, setup

setup(
    name='flyingaussian',
    version='0.1.0',
    author='Quan2630345594',
    license="BSD 3-Clause",
    packages=find_namespace_packages(include=['airgym*', 'lib*']),
    author_email='',
    description='3DGS visual and mesh-collision simulator for quadrotor RL',
    install_requires=["numpy",
                    "scipy",
                    "pyyaml",
                    "pillow",
                    "imageio",
                    "ninja",
                    'matplotlib',
                    'torch==2.4.1',
                    'rospkg',
                    'gym==0.23.1',
                    'rlpx4controller',
                    'usd-core',
                    'tensorboardX',
                    'opencv-python',
                    'torchvision',
                    'trimesh'
                    ]
)
