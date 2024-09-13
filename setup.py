from setuptools import setup, find_packages

setup(
    name="voc-annotation-verifier",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
    ],
    entry_points={
        "console_scripts": [
            "voc-annotation-verifier=voc_annotation_verifier.verifier:main",
        ],
    },
)