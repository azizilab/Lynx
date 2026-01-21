#!/bin/bash

# ==============================================================================
# Unified Benchmark Runner
# ==============================================================================
# Usage: ./run_benchmarks.sh <application> <script_name> [args...]
# 
# Example: 
#   ./run_benchmarks.sh liver run_GASTON.py
#   ./run_benchmarks.sh thymus run_SIMVI.py --some-flag
#
# This script automatically detects the appropriate virtual environment 
# based on the script name and executes the python script with the 
# project root in PYTHONPATH.
# ==============================================================================

APP=$1
SCRIPT=$2
shift 2
ARGS="$@"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_ROOT="$PROJECT_ROOT/venvs"
SCRIPT_PATH="$PROJECT_ROOT/benchmarks/$APP/$SCRIPT"

# 1. Validation
if [[ -z "$APP" ]] || [[ -z "$SCRIPT" ]]; then
    echo "Usage: $0 <application> <script_name> [args...]"
    echo "Available applications: liver, breast, thymus"
    exit 1
fi

if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "Error: Script file not found: $SCRIPT_PATH"
    exit 1
fi

# 2. Environment Selection
VENV_NAME=""

case "$SCRIPT" in
    *"GASTON"*)      VENV_NAME="gaston_env" ;;
    *"Novae"*)       VENV_NAME="novae_env" ;;
    *"SpaceFlow"*)   VENV_NAME="spaceflow_env" ;;
    *"SpatialGlue"*) VENV_NAME="spatialglue_env" ;;
    *"SIMVI"*)       VENV_NAME="simvi_env" ;;
    *)               
        echo "Warning: No specific environment mapping found for $SCRIPT."
        echo "Running with the currently active environment."
        ;;
esac

# 3. Execution
echo "----------------------------------------------------------------"
echo "Benchmark: $APP / $SCRIPT"
echo "Date:      $(date)"
echo "----------------------------------------------------------------"

if [[ -n "$VENV_NAME" ]]; then
    if [[ -d "$VENV_ROOT/$VENV_NAME" ]]; then
        echo "Activating environment: $VENV_NAME"
        source "$VENV_ROOT/$VENV_NAME/bin/activate"
    else
        echo "Error: Environment '$VENV_NAME' not found in $VENV_ROOT/"
        echo "Please ensure the venv exists."
        exit 1
    fi
fi

# Ensure project root is in PYTHONPATH so imports work
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "Executing..."
python "$SCRIPT_PATH" $ARGS
EXIT_CODE=$?

if [[ -n "$VENV_NAME" ]]; then
    deactivate
fi

echo "----------------------------------------------------------------"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Success."
else
    echo "Failed with exit code $EXIT_CODE."
fi
exit $EXIT_CODE
