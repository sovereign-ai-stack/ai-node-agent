#!/usr/bin/env python3
"""
vLLM Deployment & Lifecycle Manager.
Responsible for:
1. Selecting the optimal model tier based on available VRAM.
2. Calculating Tensor Parallel (TP) size across multi-GPU setups.
3. Generating optimized Docker Compose configurations or Docker run commands.
4. Starting, monitoring, and stopping vLLM containers.
"""

import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from hardware_detector import SystemHardwareReport


class VLLMManager:
    def __init__(self, catalog_path: str):
        with open(catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

    def select_tier(
        self,
        report: SystemHardwareReport,
        forced_tier_id: Optional[str] = None,
        embedding_reserve: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], int]:
        """
        Selects the best model tier based on total or single GPU VRAM.
        Returns: (tier_dict, tensor_parallel_size)
        """
        reserve = embedding_reserve or self.catalog.get("embedding_reserve_vram_gb", 1.5)
        tiers = self.catalog["tiers"]

        if forced_tier_id:
            for t in tiers:
                if t["id"] == forced_tier_id:
                    tp = self._calculate_tp(report, t)
                    return t, tp
            raise ValueError(f"Tier with id '{forced_tier_id}' not found in catalog.")

        # If no GPUs or Docker NVIDIA runtime missing, fallback to CPU
        if not report.gpus or not report.docker_installed:
            cpu_tier = next((t for t in tiers if not t.get("requires_gpu", True)), tiers[0])
            return cpu_tier, 1

        effective_vram = max(0.0, report.total_vram_gb - reserve)
        max_single = max(0.0, report.max_single_gpu_vram_gb - reserve)

        # Sort tiers ascending by min_vram_gb
        sorted_tiers = sorted(tiers, key=lambda x: x["min_vram_gb"])
        selected_tier = sorted_tiers[0]

        for t in sorted_tiers:
            if effective_vram >= t["min_vram_gb"]:
                # Check if single GPU fits or if multi-GPU tensor parallel is required
                selected_tier = t

        tp = self._calculate_tp(report, selected_tier)
        return selected_tier, tp

    def _calculate_tp(self, report: SystemHardwareReport, tier: Dict[str, Any]) -> int:
        """Determines the Tensor Parallel size based on GPU count and tier requirements."""
        gpu_count = len(report.gpus)
        if gpu_count <= 1 or not tier.get("requires_gpu", True):
            return 1
        
        configured_tp = tier.get("tensor_parallel_size", 1)
        # Cap TP to available GPU count
        return min(gpu_count, configured_tp) if configured_tp > 1 else 1

    def build_docker_command(
        self,
        tier: Dict[str, Any],
        tp_size: int,
        port: int = 8000,
        container_name: str = "vllm-node",
        local_model_path: Optional[str] = None,
        hf_token: Optional[str] = None,
        custom_served_name: Optional[str] = None,
        extra_vllm_args: Optional[List[str]] = None,
    ) -> List[str]:
        """Constructs an optimized `docker run` command for vLLM."""
        image = self.catalog.get("default_vllm_image", "vllm/vllm-openai:latest")
        served_name = custom_served_name or tier["served_model_name"]
        model_to_load = local_model_path if local_model_path else tier["recommended_model"]

        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--restart", "unless-stopped",
            "-p", f"{port}:8000",
            "--shm-size", "16g",
            "--ipc=host",
        ]

        if tier.get("requires_gpu", True):
            cmd.extend(["--gpus", "all"])

        # Volume mounts
        if local_model_path and os.path.exists(local_model_path):
            abs_path = os.path.abspath(local_model_path)
            cmd.extend(["-v", f"{abs_path}:/models/active_model:ro"])
            model_to_load = "/models/active_model"
        else:
            cmd.extend(["-v", "vllm-cache:/root/.cache/huggingface"])

        # Environment variables
        token = hf_token or os.environ.get("HF_TOKEN")
        if token:
            cmd.extend(["-e", f"HF_TOKEN={token}"])

        # vLLM Server Arguments
        vllm_args = [
            image,
            "--model", model_to_load,
            "--served-model-name", served_name,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--gpu-memory-utilization", str(tier["gpu_memory_utilization"]),
            "--max-model-len", str(tier["max_model_len"]),
            "--max-num-seqs", str(tier["max_num_seqs"]),
        ]

        if tier.get("quantization"):
            vllm_args.extend(["--quantization", tier["quantization"]])

        if tier.get("dtype"):
            vllm_args.extend(["--dtype", tier["dtype"]])

        if tp_size > 1:
            vllm_args.extend(["--tensor-parallel-size", str(tp_size)])

        # Append tier extra args (e.g. prefix caching, cpu device)
        for arg in tier.get("extra_args", []):
            if arg not in vllm_args:
                vllm_args.append(arg)

        if extra_vllm_args:
            vllm_args.extend(extra_vllm_args)

        cmd.extend(vllm_args)
        return cmd

    def generate_docker_compose_yaml(
        self,
        tier: Dict[str, Any],
        tp_size: int,
        port: int = 8000,
        container_name: str = "vllm-node",
        local_model_path: Optional[str] = None,
        hf_token: Optional[str] = None,
        custom_served_name: Optional[str] = None,
    ) -> str:
        """Generates a standalone, reproducible docker-compose.yml for this specific node."""
        image = self.catalog.get("default_vllm_image", "vllm/vllm-openai:latest")
        served_name = custom_served_name or tier["served_model_name"]
        model_to_load = "/models/active_model" if local_model_path else tier["recommended_model"]

        args_list = [
            f"--model {model_to_load}",
            f"--served-model-name {served_name}",
            "--host 0.0.0.0",
            "--port 8000",
            f"--gpu-memory-utilization {tier['gpu_memory_utilization']}",
            f"--max-model-len {tier['max_model_len']}",
            f"--max-num-seqs {tier['max_num_seqs']}",
        ]

        if tier.get("quantization"):
            args_list.append(f"--quantization {tier['quantization']}")
        if tier.get("dtype"):
            args_list.append(f"--dtype {tier['dtype']}")
        if tp_size > 1:
            args_list.append(f"--tensor-parallel-size {tp_size}")
        for extra in tier.get("extra_args", []):
            args_list.append(extra)

        joined_args = " ".join(args_list)

        gpu_section = """
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]""" if tier.get("requires_gpu", True) else ""

        volumes_section = f"""
    volumes:
      - {os.path.abspath(local_model_path)}:/models/active_model:ro""" if (local_model_path and os.path.exists(local_model_path)) else """
    volumes:
      - vllm-hf-cache:/root/.cache/huggingface"""

        named_volumes = "" if (local_model_path and os.path.exists(local_model_path)) else """
volumes:
  vllm-hf-cache:"""

        env_section = f"""
    environment:
      - HF_TOKEN={hf_token or '${HF_TOKEN:-}'}""" if hf_token or os.environ.get("HF_TOKEN") else ""

        compose_content = f"""version: '3.8'

services:
  {container_name}:
    image: {image}
    container_name: {container_name}
    restart: unless-stopped
    shm_size: '16gb'
    ipc: host
    ports:
      - "{port}:8000"{env_section}{volumes_section}{gpu_section}
    command: >
      {joined_args}
{named_volumes}
"""
        return compose_content
