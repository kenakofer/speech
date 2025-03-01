#!/bin/bash
# Run all tests with coverage

# Install test dependencies if needed
if ! command -v pytest &> /dev/null; then
    echo "Installing pytest and coverage..."
    pip install pytest pytest-cov
fi

# Run the tests
pytest