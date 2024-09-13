from setuptools import setup, find_packages

setup(
    name="annotation-verifier-cv",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
    ],
    entry_points={
        "console_scripts": [
            "annotation-verifier-cv=annotation_verifier_cv:run_annotation_verifier",
        ],
    },
)