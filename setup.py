from setuptools import setup, find_packages


with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.19.0", "matplotlib>=3.2.0"]

setup(
    name="mlaxe",
    version="0.0.13",
    author="axe980703",
    description="Machine Learning library with visualization support",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/axe980703/mlaxe",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    install_requires=requirements
)
