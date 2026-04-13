from setuptools import find_packages, setup


setup(
    name="tdc-kv",
    version="0.1.0",
    description="Tier-based KV cache compression with dependency-aware chunk scoring",
    python_requires=">=3.10",
    packages=find_packages(include=["src", "src.*", "benchmarks", "benchmarks.*"]),
    include_package_data=True,
    install_requires=[
        "torch>=2.3.0",
        "numpy>=2.0.0",
        "transformers>=4.45.0",
        "matplotlib>=3.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=9.0.0",
            "pytest-cov>=5.0.0",
            "ruff>=0.6.0",
        ]
    },
)
