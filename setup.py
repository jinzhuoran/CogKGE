from setuptools import setup
import setuptools

with open("requirements.txt",encoding="utf-8") as f:
    reqs = f.read()
pkgs=[]
setup(
    name="cogkge",
    version="0.2.0",
    description="CogKGE: A Knowledge Graph Embedding Toolkit and Benckmark for Representing Multi-source and Heterogeneous Knowledge",
    url="https://github.com/jinzhuoran/CogKGE/",
    author="CogNLP Team",
    packages=setuptools.find_packages(),
    author_email="zhuoran.jin@nlpr.ia.ac.cn",
    license='MIT',
    install_requires=reqs.strip().split("\n"),
    python_requires=">=3.7,<3.9",
    zip_safe=False,
)
