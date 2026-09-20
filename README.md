# Sovereign AI Node Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![vLLM](https://img.shields.io/badge/vLLM-0.6.0-blue.svg)](https://github.com/vllm-project/vllm)
[![Ray](https://img.shields.io/badge/Ray-Distributed-009688.svg?style=flat)](https://www.ray.io/)

Sovereign AI Node Agent is a headless, high-throughput distributed inference engine for self-hosted LLM deployments. It virtualizes consumer-grade GPUs into a unified computing cluster, connects securely to the Central Control Plane over a zero-trust Mesh network, and executes LLM generation using PagedAttention and Continuous Batching.

The repository contains the vLLM wrapper, Ray orchestration scripts, hardware discovery protocols, and Mesh networking integration. It contains no embedded model weights or central API logic.

## Features

- **High-Throughput Inference:** Achieves massive token-per-second rates via \LLM\, optimized for AWQ/GPTQ quantization on constrained VRAM.
- **Distributed GPU Clustering:** Natively supports multi-GPU and multi-node clustering using \Ray\.
- **Pipeline Parallelism (PP):** Distributes transformer layers across physically distinct servers in the Mesh network.
- **Tensor Parallelism (TP):** Slices massive weight matrices across local GPUs for ultra-low latency.
- **Secure Mesh Registration:** Joins the Sovereign AI ecosystem via Tailscale without exposing public IPs or open internet ports.
- **Resilient Hardware Discovery:** Four-tier fallback hardware detection (\NVML\ -> \
vidia-smi\ -> \PyTorch CUDA\ -> \WMI\) for robust initialization.

## Requirements

- Git
- Python 3.10 or newer (Linux / WSL2 highly recommended)
- NVIDIA GPU with compatible CUDA Toolkit
- [uv](https://docs.astral.sh/uv/) or \pip\
- Tailscale (or WireGuard) configured for Mesh node connections

## Quick start

Clone the repository and install the dependencies on your worker node:

\\\ash
git clone https://github.com/sovereign-ai-stack/ai-node-agent.git
cd ai-node-agent

# Install dependencies (vLLM and Ray)
pip install -r requirements.txt
\\\

Configure your \.env\ pointing to the Central Control Plane:

\\\env
CENTRAL_PLANE_IP=100.x.x.x
LITELLM_API_KEY=sk-sovereign-master
MODEL_PATH=/models/Qwen2.5-3B-Instruct-AWQ
MAX_SEQ_LEN=1024
\\\

Start the node in standard single-GPU mode:

\\\ash
python api_utils.py --model_tag Qwen/Qwen2.5-3B-Instruct-AWQ --quantization awq --gpu_memory_utilization 0.5
\\\

## Distributed Parallelism

The agent natively supports slicing models across multiple distinct hardware nodes using Ray.

To initialize Pipeline Parallelism across two separate machines:

1. On **Machine A**, start the Ray Head:
   \\\ash
   ray start --head --port=6379
   \\\

2. On **Machine B**, connect to the Head via the Mesh network:
   \\\ash
   ray start --address='<Machine_A_Tailscale_IP>:6379'
   \\\

3. On **Machine A**, launch the vLLM agent with parallelism enabled:
   \\\ash
   python api_utils.py --pipeline-parallel-size 2
   \\\

The agent automatically distributes the transformer layers across the network.

## Secure Registration

Upon successful initialization, the agent securely handshakes with the Central Control Plane. It registers its model capabilities and hardware limits automatically over the private network.

![Node Registration Workflow](./assets/node-registration.png)

No external API exposure is required.
