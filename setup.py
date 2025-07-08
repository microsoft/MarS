from setuptools import setup

setup(
    name="market_simulation",
    version="0.0.1",
    packages=["market_simulation", "mlib"],
    description="An awesome package that does something",
    install_requires=[
        "pandas==2.2.3",
        "seaborn==0.13.2",
        "ruamel.yaml==0.18.6",
        "scipy==1.15.2",
        "pydantic==2.10.3",
        "python-dotenv==1.0.1",
        "rich==13.9.4",
        "zstandard==0.23.0",
        "pydantic-settings==2.8.1",
        "ray[serve]==2.44.0",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        "transformers==4.52.1",
        "streamlit==1.40.1",
    ],
    extras_require={
        "dev": [
            "mypy==1.15.0",
            "pre-commit==4.2.0",
            "types-protobuf==5.29.1.20250315",
            "types-requests==2.32.0.20250306",
            "tqdm-stubs==0.2.1",
            "ipython==9.0.2",
            "ruff==0.11.2",
            "pandas-stubs==2.2.3.250308",
        ],
    },
)
