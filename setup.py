"""
Package setup for local development.
Install in editable mode: pip install -e .
"""
from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="news-sentiment-analysis",
    version="1.0.0",
    description=(
        "Multi-class news headline sentiment analysis "
        "using BiLSTM with Self-Attention"
    ),
    author="Karthik Mulugu",
    author_email="karthikmulugu14@gmail.com",
    url="https://github.com/Karthik0809/news-sentiment-bilstm",
    packages=find_packages(exclude=["tests*", "notebooks*", "research*"]),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
