#!/bin/bash
# This 'set -e' command makes the script exit immediately if any command fails.
set -e

echo "--- 1. Updating system packages and installing nano ---"
apt-get update
apt-get install -y nano

echo "--- 2. Installing uv package manager (globally) ---"
pip install uv

echo "--- 3. Creating virtual environment in ./.venv ---"
uv venv

echo "--- 4. Installing project dependencies from pyproject.toml ---"
# 'uv pip install .' is the standard command to install a project
# defined by pyproject.toml into the active virtual environment.
# Your original command '-r pyproject.toml' is not the typical way to do this.
uv pip install .

# --- Optional: ---
# If you need an *editable* install (so changes to your code are
# reflected without reinstalling), use this command instead of the one above:
# uv pip install -e .

echo "--- Setup Complete! Environment is ready. ---"
