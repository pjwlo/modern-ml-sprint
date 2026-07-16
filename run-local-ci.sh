#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=== Running Local Python CI Pipeline ==="

echo "Step 1: Checking code formatting (Black)..."
black .

echo "Step 2: Running Linter (Flake8)..."
# We exclude our virtual environment so we don't lint third-party libraries
flake8 . --exclude=.venv

echo "Step 3: Running Test Suite (Pytest)..."
pytest

echo "================================="
echo "  Local CI Passed Successfully!  "
echo "================================="
