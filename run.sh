#!/usr/bin/env bash
set -e

GATEWAY_URL=${1:-"http://localhost:8200"}

echo "=================================================="
echo " Starting Enterprise AI Node Agent..."
echo " Central Gateway: $GATEWAY_URL"
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found! Please install Python 3.8+."
    exit 1
fi

# Run Node Agent
python3 node_agent.py --gateway-url "$GATEWAY_URL"
