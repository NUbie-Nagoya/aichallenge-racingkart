#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
cd "${SCRIPT_DIR}" || exit

# Create the virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating shared virtual environment..."
    python3 -m venv .venv
fi

# EXPLICITLY activate the environment to block system Python access
source .venv/bin/activate

# Define the packages required for the generator
REQUIRED_PACKAGES="opencv-python pyyaml numpy<2 pandas scipy casadi tabulate"
MISSING_PACKAGES=""

# Check the shared .venv for missing dependencies
for pkg in $REQUIRED_PACKAGES; do
    if ! pip show "$pkg" > /dev/null 2>&1; then
        MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
    fi
done

# Install only the missing packages
if [ -n "$MISSING_PACKAGES" ]; then
    echo "Installing missing dependencies:$MISSING_PACKAGES"
    pip install $MISSING_PACKAGES
fi

# Run the generator script
python3 generate_track.py "$@"