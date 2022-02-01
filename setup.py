### SETUP FOR GTA

import sys
from setuptools import setup, find_packages

setup(
    name = "gta",
    version = __version__,
    packages = find_packages(),
    license = 'MIT',
    description="Generated Textured Avatars (GTA)",
    author = 'Maxime Raafat',
    author_email = 'raafatm@student.ethz.ch',
    url = 'https://github.com/maximeraafat/gta'
    install_requires = [
        'numpy',
        'torch',
        'torchvision',
        'scipy',
        'pillow',
        'tqdm',
        'opencv-python',
        'matplotlib',
        'fire',
        'retry',
        'einops>=0.3',
        'kornia==0.5.4',
        'adabelief-pytorch'
    ],
)
