import numpy
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize('native_utils.pyx'),
    include_dirs=[numpy.get_include()], install_requires=['cython', 'numpy', 'tqdm', 'scipy', 'cvxopt']
)
