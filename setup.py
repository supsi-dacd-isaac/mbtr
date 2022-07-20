import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mbtr",
    version="0.1.2",
    author="Lorenzo Nespoli",
    author_email="lorenzo.nespoli@supsi.ch",
    description="Multivariate Boosted Trees Regressor package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/supsi-dacd-isaac/mbtr",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['matplotlib>=3.1.2',
        'numpy>=1.18.1',
        'networkx>=2.4',
        'tqdm>=4.41.1',
        'numba>=0.47.0',
        'scipy>=1.4.1',
        'twisted>=20.3.0',
        'requests>=2.22.0',
        'lightgbm>=2.3.1',
        'pandas>=1.0.3',
        'seaborn>=0.10.1'
        ],
    python_requires='>=3.7',
)
