#!/usr/bin/env python3
"""
vLLM Deployment & Lifecycle Manager (Enterprise Smart Edition).
Responsible for:
1. Selecting the optimal model tier based on available VRAM and benchmark ratings.
2. Recommending Top 3 models tailored to hardware (General QA, Deep Reasoning, Coding).
3. Scanning local storage (./models, D:/models, HF cache) for existing weights.
4. Generating optimized Docker commands / compose configurations.
5. Starting, monitoring, and managing vLLM containers.
"""

import glob
import json
import os
import subprocess
import sys
from pathlib import Path
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
        Selects the best model tier based on total GPU VRAM.
        Returns: (tier_dict, tensor_parallel_size)
        """
        reserve = embedding_reserve or self.catalog.get("embedding_reserve_vram_gb", 0.5)
        tiers = self.catalog["tiers"]

        if forced_tier_id:
            for t in tiers:
                if t["id"] == forced_tier_id:
                    tp = self._calculate_tp(report, t)
                    return t, tp
            raise ValueError(f"Tier with id '{forced_tier_id}' not found in catalog.")

        # If no GPUs or Docker NVIDIA runtime missing, fallback to CPU
        if not report.gpus or not report.docker_installed or report.total_vram_gb <= 0:
            cpu_tier = tiers[0]
            return cpu_tier, 1

        effective_vram = max(0.0, report.total_vram_gb - reserve)

        # Sort tiers ascending by min_vram_gb
        sorted_tiers = sorted(tiers, key=lambda x: x["min_vram_gb"])
        selected_tier = sorted_tiers[0]

        for t in sorted_tiers:
            if effective_vram >= t["min_vram_gb"]:
                selected_tier = t

        tp = self._calculate_tp(report, selected_tier)
        return selected_tier, tp

    def get_top_model_recommendations(
        self,
        tier: Dict[str, Any],
        custom_search_dirs: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts and enriches top recommended models for the chosen tier.
        Checks if each model is already available locally on disk.
        """
        models = tier.get("recommended_models", [])
        if not models:
            # Fallback legacy catalog structure
            return [{
                "model_id": tier.get("recommended_model", "Qwen/Qwen2.5-3B-Instruct"),
                "local_names": [tier.get("served_model_name", "qwen-3b")],
                "served_model_name": tier.get("served_model_name", "qwen-3b"),
                "category": "general",
                "category_label": "مدل پیش‌فرض عمومی و پایگاه دانش",
                "benchmark_score": "Standard",
                "supported_roles": tier.get("supported_roles", ["general-model"]),
                "dtype": tier.get("dtype", "float16"),
                "quantization": tier.get("quantization"),
                "max_model_len": tier.get("max_model_len", 8192),
                "max_num_seqs": tier.get("max_num_seqs", 24),
                "gpu_memory_utilization": tier.get("gpu_memory_utilization", 0.90),
                "description": tier.get("description", ""),
                "local_path": self.find_local_model_weights(tier.get("recommended_model", ""), search_dirs=custom_search_dirs),
            }]

        enriched = []
        for m in models:
            m_copy = dict(m)
            local_path = self.find_local_model_weights(
                m["model_id"],
                aliases=m.get("local_names", []),
                search_dirs=custom_search_dirs,
            )
            m_copy["local_path"] = local_path
            m_copy["is_local"] = bool(local_path)
            enriched.append(m_copy)

        return enriched

    def _calculate_tp(self, report: SystemHardwareReport, tier: Dict[str, Any]) -> int:
        """Determines the Tensor Parallel size based on GPU count and tier requirements."""
        gpu_count = len(report.gpus)
        if gpu_count <= 1:
            return 1
        
        configured_tp = tier.get("tensor_parallel_size", 1)
        return min(gpu_count, configured_tp) if configured_tp > 1 else 1

    # ----------------------------------------------------------------------
    # Smart Local Resource Verification (Images, Archives & Model Weights)
    # ----------------------------------------------------------------------

    def is_docker_image_present(self, image_name: str) -> bool:
        """Checks if a Docker image is already available in the local Docker daemon."""
        try:
            res = subprocess.run(
                ["docker", "image", "inspect", image_name],
                capture_output=True,
                text=True,
                check=False,
            )
            return res.returncode == 0
        except Exception:
            return False

    def find_local_tar_archives(self, search_dirs: Optional[List[str]] = None) -> List[str]:
        """Scans current and common directories for saved Docker image tar files."""
        dirs = search_dirs or [".", "..", "D:/", "D:/models", "D:/docker_images", "C:/docker_images"]
        found = []
        for d in dirs:
            if os.path.exists(d):
                for ext in ["*.tar", "*.tar.gz"]:
                    for f in glob.glob(os.path.join(d, ext)):
                        if "vllm" in os.path.basename(f).lower():
                            found.append(os.path.abspath(f))
        return list(set(found))

    def load_docker_tar_image(self, tar_path: str) -> bool:
        """Loads a Docker image archive (.tar) into Docker daemon without internet access."""
        print(f"📦 Loading Docker image from local archive: {tar_path} ...")
        try:
            res = subprocess.run(["docker", "load", "-i", tar_path])
            return res.returncode == 0
        except Exception as e:
            print(f"❌ Error loading tar archive: {e}")
            return False

    def find_local_model_weights(
        self,
        model_identifier: str,
        aliases: Optional[List[str]] = None,
        search_dirs: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Intelligently searches for local model weights on disk across all drives:
        1. Exact folder names matching aliases (e.g. `Qwen2.5-3B-Instruct`, `DeepSeek-R1-Distill-Qwen-1.5B`)
        2. Subdirectories in `./models/`, `D:/models/`, `C:/models/`
        3. Hugging Face Hub cache directory (~/.cache/huggingface/hub)
        """
        all_names = [model_identifier, model_identifier.split("/")[-1]]
        if aliases:
            all_names.extend(aliases)
        all_names = list(set(all_names))

        base_search_paths = search_dirs or [
            ".",
            "./models",
            "../models",
            "D:/",
            "D:/models",
            "C:/",
            "C:/models",
            os.path.join(os.path.expanduser("~"), "models"),
            os.path.join(os.path.expanduser("~"), "Downloads"),
            "D:/Downloads",
        ]

        candidates = []
        for bp in base_search_paths:
            if not os.path.exists(bp):
                continue
            for name in all_names:
                candidates.append(os.path.join(bp, name))
                # Also check direct children in base path
                try:
                    for child in os.listdir(bp):
                        child_path = os.path.join(bp, child)
                        if os.path.isdir(child_path) and child.lower() == name.lower():
                            candidates.append(child_path)
                except Exception:
                    pass

        # Also add HF hub cache snapshot candidate
        clean_hf_repo = f"models--{model_identifier.replace('/', '--')}"
        hf_cache_snap = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", clean_hf_repo, "snapshots")
        candidates.append(hf_cache_snap)

        for cand in candidates:
            if os.path.exists(cand):
                if "snapshots" in cand:
                    subdirs = [os.path.join(cand, s) for s in os.listdir(cand) if os.path.isdir(os.path.join(cand, s))]
                    if subdirs and self._is_valid_model_dir(subdirs[0]):
                        return os.path.abspath(subdirs[0])
                elif self._is_valid_model_dir(cand):
                    return os.path.abspath(cand)

        return None

    def _is_valid_model_dir(self, directory: str) -> bool:
        """Validates that a directory contains actual LLM model files."""
        if not os.path.isdir(directory):
            return False
        try:
            files = os.listdir(directory)
            has_config = "config.json" in files or "model.json" in files or "params.json" in files
            has_weights = any(f.endswith((".safetensors", ".bin", ".pt", ".gguf")) for f in files)
            return has_config or has_weights
        except Exception:
            return False

    # ----------------------------------------------------------------------
    # Docker Command & Compose Generation
    # ----------------------------------------------------------------------

    def build_docker_command(
        self,
        tier: Dict[str, Any],
        selected_model: Dict[str, Any],
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
        served_name = custom_served_name or selected_model.get("served_model_name") or tier.get("served_model_name", "model")
        model_to_load = local_model_path if local_model_path else selected_model["model_id"]

        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--restart", "unless-stopped",
            "-p", f"{port}:8000",
            "--shm-size", "16g",
            "--ipc=host",
        ]

        if tier.get("min_vram_gb", 0) > 0:
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

        # Utilization & Sequence Limits
        gpu_util = selected_model.get("gpu_memory_utilization") or tier.get("gpu_memory_utilization", 0.90)
        max_len = selected_model.get("max_model_len") or tier.get("max_model_len", 8192)
        max_seqs = selected_model.get("max_num_seqs") or tier.get("max_num_seqs", 24)

        vllm_args = [
            image,
            "--model", model_to_load,
            "--served-model-name", served_name,
            "--host", "0.0.0.0",
            "--port", "8000",
            "--gpu-memory-utilization", str(gpu_util),
            "--max-model-len", str(max_len),
            "--max-num-seqs", str(max_seqs),
        ]

        quant = selected_model.get("quantization") or tier.get("quantization")
        if quant:
            vllm_args.extend(["--quantization", quant])

        dtype = selected_model.get("dtype") or tier.get("dtype")
        if dtype:
            vllm_args.extend(["--dtype", dtype])

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
