from setuptools import setup, find_packages

setup(
    name="aadp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "peft>=0.10.0",
    ],
)
