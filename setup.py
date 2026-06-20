from setuptools import setup, find_packages

setup(
    name="epicon",
    version="0.3.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
    ],
    extras_require={
        "numba": ["numba>=0.56"],
        "all": ["numba>=0.56"],
    },
    python_requires=">=3.10",
    description="Lightweight from-scratch ML library — neural nets, tree-based models, linear models, and more",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/kebtes/epicon",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)