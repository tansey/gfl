"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages, Extension
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# C library for the GFL solver
module1 = Extension('libgraphfl',
                    include_dirs = ['cpp/include/'],
                    sources = ['cpp/src/graph_fl.c'])

setup(
    name='pygfl',
    version='1.0.0',
    description='A Fast and Flexible Graph-Fused Lasso Solver',
    long_description=long_description,
    url='https://github.com/tansey/gfl',
    author='Wesley Tansey',
    author_email='wes.tansey@gmail.com',
    license='LGPL',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: LGPL License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7'
    ],
    keywords='statistics machinelearning lasso fusedlasso',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy', 'scipy', 'networkx'],
    package_data={
        'pygfl': [],
    },
    entry_points={
        'console_scripts': [
            'maketrails=pygfl.trail_decomposition:main',
            'graphfl=pygfl:main'
        ],
    },
    ext_modules=[module1]
)













