from setuptools import setup

setup(
    name="cogkge",
    version="0.1",
    description="CogKGE: A Knowledge Graph Embedding Toolkit and Benckmark for Representing Multi-source and Heterogeneous Knowledge",
    url="https://github.com/jinzhuoran/CogKGE/",
    author="CogNLP Team",
    author_email="zhuoran.jin@nlpr.ia.ac.cn",
    install_requires=[
        "numpy==1.19.*",
        "torch==1.7.1",
        "pyyaml",
        "pandas",
        "argparse",
        "path",
        "ax-platform==0.1.19", "botorch==0.4.0", "gpytorch==1.4.2",
        "sqlalchemy",
        "torchviz",
        "numba==0.50.*",
    ],
    python_requires=">=3.7,<3.9",
    zip_safe=False,
)
