from setuptools import setup, find_packages

setup(
    name="aadp",
    version="0.1.0",
    packages=find_packages(),
    # Core runtime deps needed to import and run the package. Heavy/optional
    # extras (monai, radgraph, bert-score, seaborn, wandb) live in
    # requirements.txt and are imported lazily where used.
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "timm>=0.9.0",
        "einops>=0.7.0",
        "SimpleITK>=2.3.0",
        "nibabel>=5.2.0",
        "datasets>=2.18.0",
        "huggingface_hub>=0.23.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scipy>=1.12.0",
        "scikit-learn>=1.4.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.8.0",
    ],
)
