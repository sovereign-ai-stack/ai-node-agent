#!/usr/bin/env python3
"""
Enterprise vLLM Node Agent (Smart Autonomous & Local-First).
The all-in-one automation script for deploying, managing, and auto-registering
local AI nodes to the centralized gateway.

Features:
  1. Hardware Diagnostic (VRAM, CUDA, Cores, RAM).
  2. Benchmark-Driven Top 3 Model Recommendations tailored to GPU VRAM.
  3. Local-First Docker Image Discovery (Inspects Docker daemon, searches .tar archives).
  4. Local-First Model Weight Discovery (Scans ./models, D:/models, HF Cache before downloading).
  5. Interactive Selection with 10s auto-fallback to #1 (Golden Choice).
  6. Dynamic Central Gateway Auto-Registration and Heartbeat.
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
from hardware_detector import detect_hardware, print_hardware_summary, ensure_docker_running
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


def load_dotenv_file(filepath: str = ".env"):
    """Reads key=value pairs from .env and injects into os.environ if not already set."""
    paths = [filepath, os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'").strip('"')
                        if k and k not in os.environ and v:
                            os.environ[k] = v
                break
            except Exception:
                pass


def main():
    load_dotenv_file(".env")

    parser = argparse.ArgumentParser(
        description="Smart Node Agent: Hardware Probe, Benchmark Recommender, and Central Auto-Registration"
    )
    # Hardware & Model options
    parser.add_argument("--catalog", default=CATALOG_DEFAULT_PATH, help="Path to model catalog JSON")
    parser.add_argument("--tier", default=None, help="Force a specific tier ID (e.g. tier-4gb-vram)")
    parser.add_argument("--local-model-path", default=os.environ.get("LOCAL_MODEL_PATH"), help="Local directory containing model weights (air-gapped)")
    parser.add_argument("--image-tar", default=os.environ.get("IMAGE_TAR_PATH"), help="Path to a local vllm Docker image .tar archive to load")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token for gated models")
    parser.add_argument("--served-name", default=None, help="Custom served model name alias (default: from catalog)")
    
    # Deployment options
    parser.add_argument("--port", type=int, default=int(os.environ.get("VLLM_PORT", "8000")), help="Host port for vLLM container (default: 8000)")
    parser.add_argument("--container-name", default="vllm-node", help="Docker container name (default: vllm-node)")
    parser.add_argument("--dry-run", action="store_true", help="Print docker command without executing")
    parser.add_argument("--probe-only", action="store_true", help="Run hardware diagnostic and exit")
    parser.add_argument("--force-cpu", action="store_true", help="Allow running on CPU even if performance is low")
    parser.add_argument(
        "-y", "--non-interactive",
        action="store_true",
        default=os.environ.get("NON_INTERACTIVE", "").lower() in ("true", "1", "yes"),
        help="Run in non-interactive mode (auto-accept recommended model #1)"
    )
    
    # Registration & Gateway options
    parser.add_argument("--gateway-url", default=os.environ.get("GATEWAY_URL", "http://localhost:8200"), help="Central Gateway / Registry URL (e.g. http://192.168.1.100:8200)")
    parser.add_argument("--node-id", default=os.environ.get("NODE_ID"), help="Unique node identifier (default: auto-generated hostname-uuid)")
    parser.add_argument("--api-base", default=os.environ.get("NODE_API_BASE"), help="Reachable HTTP base URL for this node (default: auto-detected LAN IP)")
    parser.add_argument("--auth-token", default=os.environ.get("AUTH_TOKEN"), help="Bearer token for Central Gateway authentication")
    parser.add_argument("--heartbeat-interval", type=int, default=15, help="Heartbeat interval in seconds (default: 15)")

    args = parser.parse_args()

    # Step 1: Detect Hardware
    print("\n🔍 Phase 1: Probing System Hardware & GPU Status...")
    report = detect_hardware()
    print_hardware_summary(report)

    if args.probe_only:
        return 0

    # Step 2: Autonomous Benchmark-Driven Model Recommendations
    manager = VLLMManager(args.catalog)
    tier, tp_size = manager.select_tier(report, forced_tier_id=args.tier)
    recommendations = manager.get_top_model_recommendations(tier)

    print("\n🎯 Phase 2: Autonomous Benchmark-Based Model Selection")
    print(f"  • Matched Hardware Tier : {tier['name']}")
    print(f"  • Total Usable VRAM     : {report.total_vram_gb} GB")
    print("\n  🏆 Top 3 Recommended Models Tailored for Your Hardware:")
    print("  " + "-" * 68)

    for idx, m in enumerate(recommendations[:3], 1):
        local_badge = f"✅ Local Disk ({m['local_path']})" if m.get("is_local") else "🌐 Will download from Hugging Face"
        print(f"  [{idx}] {m['model_id']}")
        print(f"      ⭐ Role / Category : {m.get('category_label', m.get('category', 'general'))}")
        print(f"      📊 Benchmark Score : {m.get('benchmark_score', 'High')}")
        print(f"      💾 Local Status    : {local_badge}")
        print(f"      ⚡ Context & Spec  : {m.get('max_model_len', 8192)} Tokens | {m.get('dtype', 'float16')} | {m.get('description', '')}")
        print("  " + "-" * 68)

    selected_model = recommendations[0]
    if not args.non_interactive and len(recommendations) > 1:
        try:
            choice = input(f"\n  👉 Select model [1-{min(3, len(recommendations))}] (Default: 1 - Recommended): ").strip()
            if choice and choice.isdigit() and 1 <= int(choice) <= len(recommendations):
                selected_model = recommendations[int(choice) - 1]
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            return 0

    print(f"\n  ✨ Chosen Model for Deployment: {selected_model['model_id']}")
    print(f"     Served Name : {selected_model.get('served_model_name')}")
    print(f"     Roles       : {selected_model.get('supported_roles', ['general-model'])}")

    # Step 3: Check and Auto-Start Docker Engine if stopped
    docker_ready = ensure_docker_running(timeout_sec=60)
    if not docker_ready:
        print("\n❌ Error: Docker daemon is not accessible. Please start Docker manually.")
        return 1

    if tier.get("min_vram_gb", 0) > 0 and not report.nvidia_runtime_available:
        print("\n❌ Error: NVIDIA Container Toolkit is missing.")
        print("   vLLM requires the NVIDIA container runtime to access GPUs.")
        print("   Install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html")
        if not args.force_cpu:
            return 1

    # ----------------------------------------------------------------------
    # Step 4: Smart Local Resource Discovery (Docker Image & Model Weights)
    # ----------------------------------------------------------------------
    print("\n🔍 Phase 3: Checking Local Storage & Model Files (Offline-First)")
    vllm_image = manager.catalog.get("default_vllm_image", "vllm/vllm-openai:latest")

    # 4.1 Check Docker Image locally
    has_image = manager.is_docker_image_present(vllm_image)
    if has_image:
        print(f"  ✅ vLLM Docker image found in local Docker: {vllm_image}")
    else:
        print(f"  ⚠️ vLLM Docker image '{vllm_image}' is NOT present in local Docker.")
        if args.image_tar and os.path.exists(args.image_tar):
            manager.load_docker_tar_image(args.image_tar)
        else:
            local_tars = manager.find_local_tar_archives()
            if local_tars:
                print(f"  💡 Found local image archive(s) on disk:")
                for i, t in enumerate(local_tars, 1):
                    print(f"     [{i}] {t}")
                if args.non_interactive or prompt_user(f"  👉 Would you like to load '{local_tars[0]}' with docker load?"):
                    manager.load_docker_tar_image(local_tars[0])
            else:
                if not args.non_interactive:
                    user_tar = input("  👉 Enter path to .tar image file (or press Enter to pull from Docker Hub): ").strip()
                    if user_tar and os.path.exists(user_tar):
                        manager.load_docker_tar_image(user_tar)
                    else:
                        print("  🌐 Proceeding with online Docker Hub pull...")

    # 4.2 Check Model Weights locally
    resolved_model_path = args.local_model_path or selected_model.get("local_path")
    if resolved_model_path and os.path.exists(resolved_model_path):
        print(f"  ✅ Found and using local model weights: {os.path.abspath(resolved_model_path)}")
    else:
        print(f"  🌐 Model weights for '{selected_model['model_id']}' will be loaded/downloaded into HF cache.")

    # Step 5: Build and Launch Docker Container
    docker_cmd = manager.build_docker_command(
        tier=tier,
        selected_model=selected_model,
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
    host_ip = get_local_ip(args.gateway_url)
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
        served_name = args.served_name or selected_model.get("served_model_name", "model")

        agent = NodeRegistrationAgent(
            node_id=node_id,
            gateway_url=args.gateway_url,
            api_base=remote_api_base,
            model_name=selected_model["model_id"],
            served_model_name=served_name,
            supported_roles=selected_model.get("supported_roles", ["general-model"]),
            hardware_meta={
                "gpus": [asdict(g) for g in report.gpus],
                "total_vram_gb": report.total_vram_gb,
                "tensor_parallel_size": tp_size,
                "tier_id": tier["id"],
                "benchmark_score": selected_model.get("benchmark_score", "N/A"),
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
            print("  🎉 NODE RUNNING & FULLY REGISTERED TO SOVEREIGN CLUSTER!")
            print(f"  • Node ID        : {node_id}")
            print(f"  • Model Served   : {served_name} ({selected_model['model_id']})")
            print(f"  • Roles Assigned : {', '.join(selected_model.get('supported_roles', []))}")
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
