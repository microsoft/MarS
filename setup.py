from setuptools import setup

setup(
    name="market_simulation",
    version="0.0.1",
    packages=["market_simulation", "mlib"],
    description="An awesome package that does something",
    install_requires=[
        "pandas==2.0.3",
        "seaborn==0.13.2",
        "protobuf==4.23.1",
        "ruamel.yaml==0.17.21",
        "fire==0.5.0",
        "scipy==1.23.5",
        "joblib==1.3.2",
        "pydantic==2.6.1",
        "python-dotenv==1.0.1",
        "rich==13.7.0",
        "zstandard==0.20.0",
        "pydantic-settings==2.1.0",
    ],
    extras_require={
        "dev": [
            "mypy==1.8.0",
            "black==24.1.1",
            "pre-commit==3.5.0",
            "types-protobuf==4.24.0.20240129",
            "tqdm-stubs==0.2.1",
            "ipython==8.12.3",
            "ruff==0.3.7",
        ],
    },
)
