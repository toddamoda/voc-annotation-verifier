# VOC Annotation Verifier

A tool for verifying and managing annotations in Pascal VOC format.

## Installation

You can install this tool directly from GitHub:

```
pip install git+https://github.com/toddamoda/voc-annotation-verifier.git
```

## Usage

After installation, you can run the tool from the command line:

```
voc-annotation-verifier /path/to/your/dataset
```

Use the following keys to interact with the tool:
- 'k': Keep the current sample and move to the next.
- 'r': Move the current sample to a 'defective' directory.
- 'n': Skip the current sample and move to the next.
- 'q': Quit the application.