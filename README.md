<div align="center">
  <h1>⚙️ Sovereign AI: Node Agent</h1>
  <p><strong>The High-Performance Distributed Inference Engine for the Sovereign AI Ecosystem</strong></p>
</div>

---

## 🚀 Overview
The **AI Node Agent** is the raw execution muscle of the Sovereign AI infrastructure. It is designed to run completely offline, air-gapped, and distributed across multiple machines in a zero-trust Mesh Network. 

Instead of requiring monolithic multi-GPU data center servers, this agent allows you to cluster consumer-grade GPUs (via Ray and vLLM) and expose them securely to the Central Control Plane over Tailscale/WireGuard. The node handles complex logic, generating tokens while the control plane orchestrates the user interaction.

### 🎬 System Architecture Demo
<video src="./assets/demo.mp4" width="100%" controls></video>

---

## 🧠 Core Architecture & Capabilities

### 1. High-Throughput Inference (vLLM)
Powered by LLM to achieve massive token-per-second rates using **PagedAttention** and **Continuous Batching**.
- Fully optimized for quantized models (e.g., AWQ, GPTQ) to run massive models (like Qwen-2.5-3B or DeepSeek-R1) on limited VRAM.
- Allows highly constrained memory footprints using variables like gpu_memory_utilization=0.5.

### 2. Distributed GPU Clustering (Ray)
Natively supports multi-GPU and multi-node clustering without writing complex socket code.
- **Tensor Parallelism (TP):** Slice massive weight matrices across GPUs on the same motherboard for ultra-low latency.
- **Pipeline Parallelism (PP):** Distribute entire transformer layers across completely different physical servers in your mesh network.

### 3. Headless & Secure Registration
Joins the Sovereign AI Mesh Network via Tailscale. Requires no public IPs or open internet ports. The node automatically discovers the Central Control Plane and securely registers its hardware capabilities.
<br><img src="./assets/node-registration.png" width="600" alt="Node Registration Process" />

### 4. Smart Hardware Discovery
When the node starts, it executes a 4-tier hardware discovery mechanism to ensure it initializes safely even in constrained environments:
NVML -> 
vidia-smi -> PyTorch CUDA -> WMI (Windows).

---

## 🛠️ Technology Stack
- **Engine:** vLLM, PyTorch
- **Models:** HuggingFace ecosystem (Qwen-2.5, DeepSeek-R1, etc.)
- **Orchestration:** Ray (for Distributed Compute)
- **Network:** Tailscale (WireGuard)
- **Runtime:** Docker / Python 3.10+

---

## 🚀 How to Run the AI Node Agent

### Prerequisites
- NVIDIA GPU(s) with CUDA drivers installed.
- Python 3.10+ (Linux/WSL2 highly recommended for LLM and Ray).
- The central-control-plane must be running and accessible over the Mesh Network.

### 1. Installation
Clone the repository and install the required dependencies:
\\\ash
pip install -r requirements.txt
# Ensure you have vllm and ray installed specifically for your CUDA version
\\\

### 2. Configure Environment
Set up your .env file pointing to the Central Control Plane's LiteLLM and Registry instances:
\\\env
CENTRAL_PLANE_IP=100.x.x.x
LITELLM_API_KEY=sk-sovereign-master
MODEL_PATH=/models/Qwen2.5-3B-Instruct-AWQ
MAX_SEQ_LEN=1024
\\\

### 3. Start the Node
To run the node in standard single-GPU mode:
\\\ash
python api_utils.py --model_tag Qwen/Qwen2.5-3B-Instruct-AWQ --quantization awq --gpu_memory_utilization 0.5
\\\

### 4. Multi-Node Distributed Mode (Pipeline Parallelism)
If you have multiple machines running the agent:
1. Start the Ray Head node on Machine A: \ay start --head\
2. Connect Machine B to the Head: \ay start --address='<Machine A Tailscale IP>:6379'\
3. Launch \LLM\ on Machine A with Ray enabled:
\\\ash
python api_utils.py --pipeline-parallel-size 2
\\\
The node agent will automatically handle distributing the transformer layers across the network!
