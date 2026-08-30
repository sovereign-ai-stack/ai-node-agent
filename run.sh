#!/usr/bin/env bash
# ======================================================================
# Enterprise AI Node Bootstrapper (Linux Zero-Touch Bare-Metal)
# Supports Ubuntu, Debian, RHEL, CentOS GPU Servers
# ======================================================================

set -e

GATEWAY_URL=${1:-"http://localhost:8200"}

echo "======================================================================"
echo "  🚀 ENTERPRISE AI NODE BOOTSTRAPPER (Linux Bare-Metal Self-Provision)"
echo "  Central Gateway: $GATEWAY_URL"
echo "======================================================================"

# 1. Check and Auto-Install Python3 & Pip if missing on fresh Linux server
if ! command -v python3 &> /dev/null || ! command -v pip3 &> /dev/null; then
    echo "📦 Python3 / Pip not found. Installing system prerequisites..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq python3 python3-pip python3-venv curl
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3 python3-pip curl
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y python3 python3-pip curl
    fi
fi

# 2. Check and Auto-Install Docker if missing on fresh Linux server
if ! command -v docker &> /dev/null; then
    echo "🐳 Docker Engine not found. Installing official Docker daemon..."
    curl -fsSL https://get.docker.com | sh
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER" || true
fi

# 3. Check and Auto-Install NVIDIA Container Toolkit if NVIDIA GPU detected
if command -v nvidia-smi &> /dev/null; then
    if ! docker info 2>/dev/null | grep -i nvidia > /dev/null; then
        echo "🔌 NVIDIA GPU detected, but NVIDIA Container Toolkit is missing. Installing..."
        if command -v apt-get &> /dev/null; then
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg --yes 2>/dev/null || true
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
            sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit
            sudo nvidia-ctk runtime configure --runtime=docker > /dev/null
            sudo systemctl restart docker
            echo "✅ NVIDIA Container Toolkit configured successfully!"
        fi
    fi
fi

# 4. Install Python Dependencies
echo "📦 Checking and installing Python dependencies..."
python3 -m pip install --upgrade pip -q 2>/dev/null || true
python3 -m pip install -r requirements.txt -q --break-system-packages 2>/dev/null || python3 -m pip install -r requirements.txt -q

# 5. Launch Node Agent
echo "🚀 Launching Node Agent..."
exec python3 node_agent.py --gateway-url "$GATEWAY_URL" "${@:2}"
