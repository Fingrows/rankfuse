from setuptools import setup, find_packages

setup(
    name="rankfuse",
    version="0.1.0",
    description="Reranking and result fusion for search and RAG pipelines",
    author="chu2bard",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "sentence-transformers>=2.2.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
    ],
)
