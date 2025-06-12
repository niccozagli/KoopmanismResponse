#!/bin/bash

# Exit immediately on error
set -e

echo "🔧 Setting up the project..."

# Install project dependencies
echo "📦 Installing dependencies using Poetry..."
poetry install

# Ensure pre-commit is in dev dependencies
if ! poetry show pre-commit &> /dev/null; then
    echo "Adding pre-commit as a development dependency..."
    poetry add --dev pre-commit
fi

# Install pre-commit hooks
echo "🧩 Installing pre-commit hooks..."
poetry run pre-commit install

# Run pre-commit on all files
echo "✅ Running pre-commit on all files..."
poetry run pre-commit run --all-files

echo "🎉 Setup complete!"
