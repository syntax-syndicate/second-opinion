#!/bin/bash

# Second Opinion MCP Server Runner
# This script ensures the MCP server runs with python3 and proper dependencies

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not installed." >&2
    exit 1
fi

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found in $SCRIPT_DIR" >&2
    exit 1
fi

# Check if requirements are installed (optional but recommended)
if [ -f "requirements.txt" ]; then
    echo "Installing/updating requirements..." >&2
    python3 -m pip install -r requirements.txt
fi

# Run the MCP server
exec python3 main.py "$@"
