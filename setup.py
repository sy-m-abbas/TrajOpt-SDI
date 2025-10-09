"""
Setup script for trajectory_opt_with_contact package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="trajectory_opt_with_contact",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Differentiable contact-implicit trajectory optimization for planar manipulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JPark-0624/trajectory_opt_with_contact.git",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
        "cvxpy>=1.2.0",
        "cvxpylayers>=0.1.5",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
)