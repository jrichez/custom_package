import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="custom",
    version="0.0.6",
    author="Jessy Richez",
    author_email="richez06@hotmail.fr",
    packages=['custom'],
    description="Custom functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jrichez/custom_package",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['sklearn']
)
