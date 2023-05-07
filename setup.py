from setuptools import find_packages, setup

setup(
    name='scrainbow',
    version='0.0.2',
    keywords=('pip','RAINBOW','single-sell'),
    description='RAINBOW: accurate cell type annotation method via contrastive learning and reference guidance for scCAS data',
    long_description="RAINBOW provides an accurate and efficient way to automatically annotate celltypes in scCAS datasets. All RAINBOW wheels distributed on PyPI are MIT licensed.",
    license='MIT License',
    url='https://github.com/BioX-NKU/RAINBOW',
    author='Siyu Li',
    packages=find_packages(),
    python_requires='>3.6.0',    
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Bio-Informatics'    ],
     install_requires=[
        'numpy>=1.22.4',
        'pandas>=1.4.3',
        'scipy>=1.9.0',
        'scikit-learn>=1.1.2',
        'numba>=0.55.2',
        'scanpy>=1.9.1',
        'matplotlib==3.5.3',
        'anndata>=0.8.0',
        'episcanpy>=0.3.2',
        'torch>=1.11.0',
       
    ]
)