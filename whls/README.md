# Coral Wheel Files

Place Coral-specific wheel files here for use in Docker builds.

## Required Files

- `pycoral-*.whl` - PyCoral API wheel (e.g., `pycoral-2.0.0-cp39-cp39-linux_x86_64.whl`)
- `tflite_runtime-*.whl` - TensorFlow Lite runtime wheel (e.g., `tflite_runtime-2.5.0.post1-cp39-cp39-linux_x86_64.whl`)

## Usage

1. Download the appropriate wheel files for your Python version and architecture
2. Place them in this `whls/` directory
3. The Dockerfile will automatically detect and use them during the build

If wheel files are not present, the Dockerfile will fall back to installing from Debian packages via apt-get.

## Download Links

- PyCoral wheels: Check [Coral documentation](https://coral.ai/software/) for download links
- TensorFlow Lite runtime: Available from [TensorFlow Lite Python quickstart](https://www.tensorflow.org/lite/guide/python)

## Note

These wheel files are typically architecture-specific (linux_x86_64) and Python version-specific (cp39 for Python 3.9, cp310 for Python 3.10, etc.).

