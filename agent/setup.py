import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reinforcebotagent",
    version="0.0.3",
    author="Will",
    author_email="contact@novucs.net",
    description="The ReinforceBot agent shared library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/novucs/project",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
