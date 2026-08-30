#!/usr/bin/env python3
"""
Enterprise vLLM Node Agent (Smart Autonomous & Local-First).
The all-in-one automation script for deploying, managing, and auto-registering
local AI nodes to the centralized gateway.

Features:
  1. Hardware Diagnostic (VRAM, CUDA, Cores, RAM).
  2. Local-First Docker Image Discovery (Inspects Docker daemon, searches .tar archives).
  3. Local-First Model Weight Discovery (Scans ./models, D:/models, HF Cache before downloading).
  4. Interactive confirmation & fallback to online downloads.
  5. Dynamic Central Gateway Auto-Registration and Heartbeat.

Usage Examples:
  # 1. Smart Interactive Boot:
  python node_agent.py --gateway-url http://localhost:8200

  # 2. Fully non-interactive / unattended mode:
  python node_agent.py --gateway-url http://localhost:8200 --non-interactive

  # 3. Direct local model & local tar:
  python node_agent.py --gateway-url http://localhost:8200 --local-model-path D:/models/qwen7b --image-tar D:/vllm.tar

  # 4. Diagnostic only:
  python node_agent.py --probe-only
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict

# Ensure UTF-8 output encoding across Windows/Linux terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from auto_register import (
    NodeRegistrationAgent,
    get_local_ip,
    wait_for_vllm_ready,
)
from hardware_detector import detect_hardware, print_hardware_summary
from vllm_manager import VLLMManager

CATALOG_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_catalog.json")


def prompt_user(question: str, default: str = "y") -> bool:
    """Helper for user confirmation prompts."""
    try:
        ans = input(f"{question} [{default.upper()}/{('n' if default == 'y' else 'y')}]: ").strip().lower()
        if not ans:
            ans = default.lower()
        return ans in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Smart Node Agent: Hardware Probe, Local-First vLLM Deployer, and Central Auto-Registration"
    )
    # Hardware & Model options
    parser.add_argument("--catalog", default=CATALOG_DEFAULT_PATH, help="Path to model catalog JSON")
    parser.add_argument("--tier", default=None, help="Force a specific tier ID (e.g. tier-8gb-vram)")
    parser.add_argument("--local-model-path", default=None, help="Local directory containing model weights (air-gapped)")
    parser.add_argument("--image-tar", default=None, help="Path to a local vllm Docker image .tar archive to load")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for gated models")
    parser.add_argument("--served-name", default=None, help="Custom served model name alias (default: from catalog)")
    
    # Deployment options
    parser.add_argument("--port", type=int, default=8000, help="Host port for vLLM container (default: 8000)")
    parser.add_argument("--container-name", default="vllm-node", help="Docker container name (default: vllm-node)")
    parser.add_argument("--export-compose", default=None, help="Export a docker-compose.node.yml file and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print docker command without executing")
    parser.add_argument("--probe-only", action="store_true", help="Run hardware diagnostic and exit")
    parser.add_argument("--force-cpu", action="store_true", help="Allow running on CPU even if performance is low")
    parser.add_argument("-y", "--non-interactive", action="store_true", help="Run in non-interactive mode (auto-accept prompts)")
    
    # Registration & Gateway options
    parser.add_argument("--gateway-url", default=None, help="Central Gateway / Registry URL (e.g. http://192.168.1.100:8200)")
    parser.add_argument("--node-id", default=None, help="Unique node identifier (default: auto-generated hostname-uuid)")
    parser.add_argument("--api-base", default=None, help="Reachable HTTP base URL for this node (default: auto-detected LAN IP)")
    parser.add_argument("--auth-token", default=None, help="Bearer token for Central Gateway authentication")
    parser.add_argument("--heartbeat-interval", type=int, default=15, help="Heartbeat interval in seconds (default: 15)")

    args = parser.parse_args()

    # Step 1: Detect Hardware
    print("\n🔍 Phase 1: Probing System Hardware & GPU Status...")
    report = detect_hardware()
    print_hardware_summary(report)

    if args.probe_only:
        return 0

    # Step 2: Select Model Tier & Calculate Tensor Parallelism
    manager = VLLMManager(args.catalog)
    tier, tp_size = manager.select_tier(report, forced_tier_id=args.tier)

    print("\n🎯 Phase 2: Autonomous Model & Optimization Selection")
    print(f"  • Selected Tier       : {tier['name']} ({tier['id']})")
    print(f"  • Recommended Model   : {tier['recommended_model']}")
    print(f"  • Quantization Engine : {tier.get('quantization') or 'None (Standard Precision)'}")
    print(f"  • Context Length      : {tier['max_model_len']} tokens")
    print(f"  • Tensor Parallel Size: {tp_size} GPU(s)")
    print(f"  • Memory Utilization  : {int(tier['gpu_memory_utilization'] * 100)}% of VRAM")

    # If export docker compose requested
    if args.export_compose:
        compose_str = manager.generate_docker_compose_yaml(
            tier=tier,
            tp_size=tp_size,
            port=args.port,
            container_name=args.container_name,
            local_model_path=args.local_model_path,
            hf_token=args.hf_token,
            custom_served_name=args.served_name,
        )
        with open(args.export_compose, "w", encoding="utf-8") as f:
            f.write(compose_str)
        print(f"\n✅ Standalone Docker Compose exported to: {os.path.abspath(args.export_compose)}")
        return 0

    # Step 3: Check Docker Requirements
    if not report.docker_installed:
        print("\n❌ Error: Docker is not installed or the Docker daemon is not running.")
        return 1

    if tier.get("requires_gpu", True) and not report.nvidia_runtime_available:
        print("\n❌ Error: NVIDIA Container Toolkit is missing.")
        print("   vLLM requires the NVIDIA container runtime to access GPUs.")
        print("   Install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html")
        if not args.force_cpu:
            return 1

    # ----------------------------------------------------------------------
    # Step 4: Smart Local Resource Discovery (Docker Image & Model Weights)
    # ----------------------------------------------------------------------
    print("\n🔍 Phase 3: Checking Local Resources (Offline-First)")
    vllm_image = manager.catalog.get("default_vllm_image", "vllm/vllm-openai:latest")

    # 4.1 Check Docker Image locally
    has_image = manager.is_docker_image_present(vllm_image)
    if has_image:
        print(f"  ✅ vLLM Docker image found in local Docker: {vllm_image}")
    else:
        print(f"  ⚠️ vLLM Docker image '{vllm_image}' is NOT present in local Docker.")
        
        # Check if user passed an explicit .tar file
        if args.image_tar and os.path.exists(args.image_tar):
            manager.load_docker_tar_image(args.image_tar)
        else:
            # Auto-search nearby directories for .tar files
            local_tars = manager.find_local_tar_archives()
            if local_tars:
                print(f"  💡 Found local image archive(s) on disk:")
                for i, t in enumerate(local_tars, 1):
                    print(f"     [{i}] {t}")
                if args.non_interactive or prompt_user(f"  👉 Would you like to load '{local_tars[0]}' with docker load?"):
                    manager.load_docker_tar_image(local_tars[0])
            else:
                if not args.non_interactive:
                    print("  💡 If you have a saved 'vllm-image.tar' file, you can load it now.")
                    user_tar = input("  👉 Enter path to .tar image file (or press Enter to pull from Docker Hub): ").strip()
                    if user_tar and os.path.exists(user_tar):
                        manager.load_docker_tar_image(user_tar)
                    else:
                        print("  🌐 Proceeding with online Docker Hub pull...")

    # 4.2 Check Model Weights locally
    resolved_model_path = args.local_model_path
    if resolved_model_path and os.path.exists(resolved_model_path):
        print(f"  ✅ Using specified local model directory: {os.path.abspath(resolved_model_path)}")
    else:
        # Auto-search on disk for the recommended model
        detected_model_dir = manager.find_local_model_weights(tier["recommended_model"])
        if detected_model_dir:
            print(f"  ✅ Found local model weights for '{tier['recommended_model']}':")
            print(f"     📁 Path: {detected_model_dir}")
            resolved_model_path = detected_model_dir
        else:
            print(f"  ⚠️ Model weights for '{tier['recommended_model']}' not found in local cache.")
            if not args.non_interactive:
                user_mpath = input(f"  👉 Enter local path to model weights (or press Enter to auto-download from Hugging Face): ").strip()
                if user_mpath and os.path.exists(user_mpath):
                    resolved_model_path = user_mpath
                    print(f"  ✅ Using local model directory: {os.path.abspath(resolved_model_path)}")
                else:
                    print(f"  🌐 Will auto-download '{tier['recommended_model']}' from Hugging Face into cache.")

    # Step 5: Build and Launch Docker Container
    docker_cmd = manager.build_docker_command(
        tier=tier,
        tp_size=tp_size,
        port=args.port,
        container_name=args.container_name,
        local_model_path=resolved_model_path,
        hf_token=args.hf_token,
        custom_served_name=args.served_name,
    )

    print("\n🐳 Phase 4: Launching vLLM Engine Container")
    print("  Command:")
    print("  " + " ".join(docker_cmd))

    if args.dry_run:
        print("\n[Dry Run Mode]: Command printed above. Exiting.")
        return 0

    # Remove any existing container with the same name if stopped/leftover
    subprocess.run(["docker", "rm", "-f", args.container_name], capture_output=True)

    print("\n⏳ Starting container...")
    proc = subprocess.run(docker_cmd)
    if proc.returncode != 0:
        print(f"\n❌ Docker launch failed with exit code {proc.returncode}")
        return proc.returncode

    # Step 6: Wait for Health / Model Ready
    host_ip = get_local_ip()
    local_api_base = f"http://localhost:{args.port}"
    remote_api_base = args.api_base or f"http://{host_ip}:{args.port}"

    print(f"\n⏳ Phase 5: Validating vLLM Engine Initialization at {local_api_base} ...")
    is_ready = wait_for_vllm_ready(local_api_base, timeout_sec=400, interval_sec=5)
    if not is_ready:
        print("❌ Error: vLLM did not reach healthy state. Check docker logs:")
        print(f"   docker logs {args.container_name}")
        return 1

    print(f"✅ vLLM is operational and listening on {remote_api_base}")

    # Step 7: Central Auto-Registration & Heartbeat Daemon
    if args.gateway_url:
        print(f"\n🌐 Phase 6: Autonomous Registration with Central Gateway ({args.gateway_url})")
        node_id = args.node_id or f"node-{os.uname().nodename if hasattr(os, 'uname') else 'host'}-{uuid.uuid4().hex[:6]}"
        served_name = args.served_name or tier["served_model_name"]

        agent = NodeRegistrationAgent(
            node_id=node_id,
            gateway_url=args.gateway_url,
            api_base=remote_api_base,
            model_name=tier["recommended_model"],
            served_model_name=served_name,
            supported_roles=tier.get("supported_roles", []),
            hardware_meta={
                "gpus": [asdict(g) for g in report.gpus],
                "total_vram_gb": report.total_vram_gb,
                "tensor_parallel_size": tp_size,
                "tier_id": tier["id"],
            },
            heartbeat_interval=args.heartbeat_interval,
            auth_token=args.auth_token,
        )

        reg_ok = agent.register()
        if reg_ok:
            agent.start_heartbeat_daemon()

            # Handle graceful shutdown
            def handle_exit(signum, frame):
                print("\n🛑 Shutdown signal received. Deregistering from gateway...")
                agent.stop_heartbeat_daemon()
                agent.deregister()
                print("👋 Node gracefully deregistered. Exiting.")
                sys.exit(0)

            signal.signal(signal.SIGINT, handle_exit)
            signal.signal(signal.SIGTERM, handle_exit)

            print("\n" + "=" * 70)
            print("  🎉 NODE RUNNING & FULLY REGISTERED!")
            print(f"  • Node ID        : {node_id}")
            print(f"  • Model Served   : {served_name}")
            print(f"  • Endpoint       : {remote_api_base}/v1")
            print(f"  • Heartbeat Rate : Every {args.heartbeat_interval}s")
            print("  Press Ctrl+C to stop node and deregister.")
            print("=" * 70 + "\n")

            while True:
                time.sleep(1)
        else:
            print("⚠️ Auto-registration failed. vLLM is still running locally.")
    else:
        print("\n💡 Note: No --gateway-url provided. Running standalone without central registration.")
        print(f"   API available at: {local_api_base}/v1/chat/completions")

    return 0


if __name__ == "__main__":
    sys.exit(main())
