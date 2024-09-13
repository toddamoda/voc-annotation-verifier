from setuptools import setup, find_packages

setup(
    name="voc-annotation-verifier",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "PyQt5",
        "opencv-python",
    ],
    entry_points={
        "console_scripts": [
            "voc-annotation-verifier=voc_annotation_verifier.main:main",
        ],
    },
    author="Sumit",
    author_email="sumit@visailabs.com",
    description="A PyQt-based tool for verifying image annotations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/toddamoda/voc-annotation-verifier",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)