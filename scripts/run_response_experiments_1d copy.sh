#!/bin/bash

set -e
set -u

echo "Running Response Experiments..."
poetry run python scripts/LinearResponseExperiments_OneDimensionalMap.py
echo "Done."

