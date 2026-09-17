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
        with open(catalog_path, "r", encoding="utf-8-sig") as f:
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

    def get_all_catalog_models(
        self,
        custom_search_dirs: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Extracts and enriches all models across all tiers in the catalog."""
        all_models = []
        seen_ids = set()
        for t in self.catalog.get("tiers", []):
            tier_name = t.get("name", "Standard")
            tier_min_vram = t.get("min_vram_gb", 0.0)
            for m in t.get("recommended_models", []):
                mid = m["model_id"]
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)
                m_copy = dict(m)
                m_copy["tier_id"] = t.get("id")
                m_copy["tier_name"] = tier_name
                m_copy["min_vram_gb"] = tier_min_vram
                local_path = self.find_local_model_weights(
                    mid,
                    aliases=m.get("local_names", []),
                    search_dirs=custom_search_dirs,
                )
                m_copy["local_path"] = local_path
                m_copy["is_local"] = bool(local_path)
                all_models.append(m_copy)
        return all_models

    def get_top_model_recommendations(
        self,
        tier: Dict[str, Any],
        custom_search_dirs: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts, enriches, and selects the Best-in-Class model for each core role:
          1. Best General & Knowledge Base (RAG) model (Evaluated by MMLU / MT-Bench)
          2. Best Deep Reasoning & Logic model (Evaluated by MATH / CoT)
          3. Best Coding & Software Engineering model (Evaluated by HumanEval / EvalPlus)
        """
        models = tier.get("recommended_models", [])
        if not models:
            return self.get_all_catalog_models(custom_search_dirs)[:3]

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
            
            # Extract numerical benchmark value within its domain
            bench_val = m.get("benchmark_value")
            if bench_val is None:
                import re
                score_str = str(m.get("benchmark_score", ""))
                match = re.search(r"(\d+(\.\d+)?)", score_str)
                bench_val = float(match.group(1)) if match else 50.0
            m_copy["benchmark_value"] = float(bench_val)
            enriched.append(m_copy)

        # Categorize models by domain role
        generals = [m for m in enriched if m.get("category") == "general" or "general-model" in m.get("supported_roles", [])]
        reasonings = [m for m in enriched if m.get("category") == "reasoning" or "reasoning-model" in m.get("supported_roles", [])]
        codings = [m for m in enriched if m.get("category") == "coding" or "coding-model" in m.get("supported_roles", [])]

        # Sort each domain by its specific benchmark score
        generals.sort(key=lambda x: (1 if x.get("is_local") else 0, x.get("benchmark_value", 0)), reverse=True)
        reasonings.sort(key=lambda x: (1 if x.get("is_local") else 0, x.get("benchmark_value", 0)), reverse=True)
        codings.sort(key=lambda x: (1 if x.get("is_local") else 0, x.get("benchmark_value", 0)), reverse=True)

        top_3: List[Dict[str, Any]] = []
        used_ids = set()

        # Pick #1 Best General
        if generals:
            best_gen = generals[0]
            top_3.append(best_gen)
            used_ids.add(best_gen["model_id"])

        # Pick #2 Best Reasoning
        for r in reasonings:
            if r["model_id"] not in used_ids:
                top_3.append(r)
                used_ids.add(r["model_id"])
                break

        # Pick #3 Best Coding
        for c in codings:
            if c["model_id"] not in used_ids:
                top_3.append(c)
                used_ids.add(c["model_id"])
                break

        # Fill up to 3 if any category was absent
        for m in enriched:
            if len(top_3) >= 3:
                break
            if m["model_id"] not in used_ids:
                top_3.append(m)
                used_ids.add(m["model_id"])

        # Also attach the remaining models in the tier for complete access
        for m in enriched:
            if m["model_id"] not in used_ids:
                top_3.append(m)
                used_ids.add(m["model_id"])

        return top_3

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

    @staticmethod
    def get_local_model_size_gb(directory: str) -> float:
        """Calculates total size of model weights in GB."""
        if not os.path.isdir(directory):
            return 0.0
        try:
            # If the directory contains .gguf files, measure only the single active file to be loaded (preferring Q4)
            gguf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".gguf")]
            if gguf_files:
                q4_files = [f for f in gguf_files if "q4" in os.path.basename(f).lower()]
                target_gguf = q4_files[0] if q4_files else gguf_files[0]
                return round(os.path.getsize(target_gguf) / (1024**3), 2)

            total_bytes = 0
            for root, _, files in os.walk(directory):
                for f in files:
                    if f.endswith((".safetensors", ".bin", ".pt")):
                        total_bytes += os.path.getsize(os.path.join(root, f))
            return round(total_bytes / (1024**3), 2)
        except Exception:
            return 0.0

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
        free_vram_gb: Optional[float] = None,
        total_vram_gb: Optional[float] = None,
        force_eager: bool = False,
        override_max_len: Optional[int] = None,
        override_gpu_util: Optional[float] = None,
        is_cpu_mode: bool = False,
    ) -> List[str]:
        """Constructs an optimized `docker run` command for vLLM with low-VRAM & WSL2 auto-tuning."""
        image = self.catalog.get("default_vllm_image", "vllm/vllm-openai:latest")
        served_name = custom_served_name or selected_model.get("served_model_name") or tier.get("served_model_name", "model")
        model_to_load = local_model_path if local_model_path else selected_model["model_id"]

        cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "-p", f"{port}:8000",
            "--shm-size", "16g",
            "--ipc=host",
            # Mandatory for Windows Docker / WSL2 to prevent UVA (Unified Virtual Addressing) crashes
            "-e", "VLLM_USE_V2_MODEL_RUNNER=0",
        ]

        if not is_cpu_mode:
            cmd.extend(["--gpus", "all"])

        # Volume mounts
        if local_model_path and os.path.exists(local_model_path):
            abs_path = os.path.abspath(local_model_path)
            cmd.extend(["-v", f"{abs_path}:/models/active_model:ro"])
            model_to_load = "/models/active_model"
            if os.path.isdir(abs_path):
                gguf_files = [f for f in os.listdir(abs_path) if f.endswith(".gguf")]
                if gguf_files:
                    # Prefer 4-bit quantized GGUF (e.g. Q4_K_M)
                    q4_files = [f for f in gguf_files if "q4" in f.lower()]
                    target_gguf = q4_files[0] if q4_files else gguf_files[0]
                    model_to_load = f"/models/active_model/{target_gguf}"
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

        if not is_cpu_mode and total_vram_gb and total_vram_gb <= 6.0:
            # Low VRAM GPU (e.g. 4GB laptop GPU):
            # 1. Adapt gpu_memory_utilization to real free memory to avoid startup ValueError
            avail_free = free_vram_gb if free_vram_gb else (total_vram_gb - 0.8)
            safe_ratio = round(max(0.50, min(0.75, (avail_free - 0.25) / total_vram_gb)), 2)
            gpu_util = override_gpu_util or min(gpu_util, safe_ratio)

            # 2. Smart Context Window Allocation based on real model footprint
            if override_max_len:
                max_len = override_max_len
            else:
                model_size_gb = 2.0  # Default safe assumption
                if local_model_path and os.path.exists(local_model_path):
                    if os.path.isdir(local_model_path):
                        model_size_gb = VLLMManager.get_local_model_size_gb(local_model_path)
                    elif os.path.isfile(local_model_path):
                        model_size_gb = round(os.path.getsize(local_model_path) / (1024**3), 2)
                
                if model_size_gb <= 0.1:
                    model_size_gb = 2.0

                # Formula: Allocated VRAM - Model Weights - vLLM Engine Overhead (~0.4GB)
                usable_vram_for_kv = (total_vram_gb * gpu_util) - model_size_gb - 0.4
                
                if usable_vram_for_kv > 0.3:
                    # Heuristic: 1GB of KV Cache holds ~6000 tokens for efficient small models (GQA)
                    calculated_tokens = int(usable_vram_for_kv * 6000)
                    calculated_tokens = (calculated_tokens // 1024) * 1024  # Snap to 1024 boundaries
                    max_len = min(max_len, max(2048, calculated_tokens))
                    print(f"🧠 [Smart Context Allocation]: {usable_vram_for_kv:.2f} GB usable for KV-Cache -> Dynamically set max_model_len to {max_len} tokens!")
                else:
                    # Severe memory pressure: drop to 1024 to avoid OOM
                    max_len = min(max_len, 1024)
                    print(f"⚠️ [Smart Context Allocation]: Severe memory pressure! {usable_vram_for_kv:.2f} GB usable. Clamped max_model_len to {max_len} tokens to prevent OOM crash.")

            # 3. Enforce eager execution to eliminate CUDA graph memory capture
            force_eager = True
        else:
            if override_gpu_util:
                gpu_util = override_gpu_util
            if override_max_len:
                max_len = override_max_len

        # ── GGUF Detection ────────────────────────────────────────────────────
        # GGUF is a binary quantized format (used by llama.cpp / ibnsina).
        # vLLM REQUIRES --load-format gguf explicitly when loading .gguf files.
        # Without it, vLLM tries to parse the binary as UTF-8 JSON and crashes:
        #   UnicodeDecodeError: 'utf-8' codec can't decode byte 0xbb ...
        # Also: --dtype and --quantization must NOT be passed for GGUF because
        # quantization is baked into the file itself (e.g. Q4_K_M header).
        is_gguf = model_to_load.endswith(".gguf")

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

        # Must declare GGUF load format — vLLM does not auto-detect binary format
        if is_gguf:
            vllm_args.extend(["--load-format", "gguf"])

        if force_eager:
            vllm_args.append("--enforce-eager")

        quant = selected_model.get("quantization") or tier.get("quantization")
        if quant and not is_gguf:
            # AWQ/GPTQ flags are for HuggingFace safetensors only.
            # GGUF files embed their quantization in the file header — passing
            # --quantization here would conflict and cause startup errors.
            vllm_args.extend(["--quantization", quant])

        dtype = selected_model.get("dtype") or tier.get("dtype")
        if dtype and not is_gguf:
            # dtype is irrelevant for GGUF — quantization precision is in the filename
            # (e.g. Q4_K_M = 4-bit, K-quant, Medium). Passing --dtype float16 would
            # override the GGUF's native type and cause a load error.
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
