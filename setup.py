from setuptools import setup, find_packages

setup(
    name="hyperx",
    version="0.0.1",
    author="HyperX",
    description="Implementation of the Improved HyperX Activation Function",
    long_description=open("docs/README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Gaurav-Jaiswal-1/hyperx",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "tensorflow",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.6",
)
