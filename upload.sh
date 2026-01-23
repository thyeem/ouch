#!/bin/sh

echo "Running tests..."
python -m doctest -v ouch/*.py

if [ $? -eq 0 ]; then
    echo "Tests passed."

    echo "Cleaning old builds..."
    rm -rf dist
    rm -rf ouch.egg-info

    echo "Building package..."
    python -m build

    echo "Uploading to PyPI..."
    python -m twine upload dist/*
else
    echo "Tests failed. Aborting upload."
    exit 1
fi
