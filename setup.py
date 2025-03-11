from setuptools import setup, find_packages

setup(
    name="ft8_tools",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-cov',
        ],
        'dev': [
            'flake8',
            'black',
            'mypy',
        ],
    },
    author="Peng",
    author_email="your.email@example.com",
    description="FT8工具包 - 包含FT8信号的生成和解调功能",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ft8_demodulator",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.6",
) 