from setuptools import setup

setup(
    name="cogkge",
    version="0.1",
    description="CogKGE: A Knowledge Graph Embedding Toolkit and Benckmark for Representing Multi-source and Heterogeneous Knowledge",
    url="https://github.com/jinzhuoran/CogKGE/",
    author="CogNLP Team",
    author_email="zhuoran.jin@nlpr.ia.ac.cn",
    install_requires=[
        "torch==1.10.1",
        "transformers==4.15.0",
        "openTSNE",
        "mongoengine",
        "tensorboard==2.7.0",
        "prettytable",
        "pandas==1.3.5",
        "prefetch-generator==1.0.1",
        "matplotlib",
    ],
    python_requires=">=3.7,<3.9",
    zip_safe=False,
)
