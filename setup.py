from setuptools import setup, find_packages

setup(
    name="mnga",
    version="0.1.0",
    description="Minimal NumPy Gradient Agents",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "gym",
    ],
    python_requires=">=3.6",
)
