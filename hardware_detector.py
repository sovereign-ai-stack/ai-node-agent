#!/usr/bin/env python3
"""
Hardware Detection & Capability Assessment Module.
Auto-probes GPU, CPU, RAM, Docker and NVIDIA Container Runtime.
Includes automatic Docker Desktop auto-start for Windows and Linux.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GPUInfo:
    index: int
    name: str
    total_vram_gb: float
    free_vram_gb: float
    compute_capability: str
    driver_version: str
    temperature_c: Optional[int] = None


@dataclass
class SystemHardwareReport:
    os_name: str
    arch: str
    cpu_model: str
    cpu_cores: int
    system_ram_gb: float
    gpus: List[GPUInfo]
    total_vram_gb: float
    max_single_gpu_vram_gb: float
    docker_installed: bool
    docker_version: str
    nvidia_runtime_available: bool
    detection_backend: str


def run_command(cmd: List[str], timeout_sec: int = 10) -> Optional[subprocess.CompletedProcess]:
    """Runs a shell command safely with timeout handling."""
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def get_system_ram_gb() -> float:
    """Detect total system RAM in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        pass

    if sys.platform.startswith("win"):
        try:
            res = run_command(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
            if res and res.returncode == 0 and res.stdout.strip().isdigit():
                return round(int(res.stdout.strip()) / (1024**3), 2)
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024**2), 2)
        except Exception:
            pass
    return 0.0


def get_cpu_info() -> Tuple[str, int]:
    """Detect CPU model name and logical core count."""
    cores = os.cpu_count() or 1
    model = platform.processor() or "Generic x86_64 / ARM"
    
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        model = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
    return model, cores


def detect_gpus_pynvml() -> Optional[List[GPUInfo]]:
    """Backend 1: Detect GPUs using pynvml (most accurate)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        driver_ver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_ver, bytes):
            driver_ver = driver_ver.decode("utf-8")

        gpus = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_gb = round(mem.total / (1024**3), 2)
            free_gb = round(mem.free / (1024**3), 2)
            
            # Temperature
            temp = None
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass

            # Compute Capability
            major, minor = 0, 0
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            except Exception:
                pass
            cc = f"{major}.{minor}" if major > 0 else "Unknown"

            gpus.append(
                GPUInfo(
                    index=i,
                    name=name,
                    total_vram_gb=total_gb,
                    free_vram_gb=free_gb,
                    compute_capability=cc,
                    driver_version=driver_ver,
                    temperature_c=temp,
                )
            )
        pynvml.nvmlShutdown()
        return gpus if gpus else None
    except Exception:
        return None


def detect_gpus_torch() -> Optional[List[GPUInfo]]:
    """Backend 2: Detect GPUs via PyTorch CUDA runtime."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        
        count = torch.cuda.device_count()
        gpus = []
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            total_gb = round(props.total_memory / (1024**3), 2)
            free_bytes = torch.cuda.mem_get_info(i)[0] if hasattr(torch.cuda, "mem_get_info") else props.total_memory
            free_gb = round(free_bytes / (1024**3), 2)

            gpus.append(
                GPUInfo(
                    index=i,
                    name=props.name,
                    total_vram_gb=total_gb,
                    free_vram_gb=free_gb,
                    compute_capability=f"{props.major}.{props.minor}",
                    driver_version="PyTorch CUDA Runtime",
                    temperature_c=None,
                )
            )
        return gpus if gpus else None
    except Exception:
        return None


def detect_gpus_nvidiasmi() -> List[GPUInfo]:
    """Backend 3: Detect GPUs via nvidia-smi CLI (fallback)."""
    res = run_command([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,driver_version,temperature.gpu",
        "--format=csv,noheader,nounits",
    ])
    gpus = []
    if res and res.returncode == 0 and res.stdout.strip():
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                idx = int(parts[0])
                name = parts[1]
                total = round(float(parts[2]) / 1024.0, 2)
                free = round(float(parts[3]) / 1024.0, 2)
                driver = parts[4] if len(parts) > 4 else "Unknown"
                temp = int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else None

                gpus.append(
                    GPUInfo(
                        index=idx,
                        name=name,
                        total_vram_gb=total,
                        free_vram_gb=free,
                        compute_capability="Auto-detected",
                        driver_version=driver,
                        temperature_c=temp,
                    )
                )
            except ValueError:
                continue
    return gpus


def check_docker_environment() -> Tuple[bool, str, bool]:
    """Check if Docker is running and if the NVIDIA container runtime is active."""
    res = run_command(["docker", "version", "--format", "{{.Server.Version}}"])
    if not res or res.returncode != 0:
        return False, "", False
    
    docker_version = res.stdout.strip()
    nvidia_runtime = False
    
    info_res = run_command(["docker", "info", "--format", "{{json .Runtimes}}"])
    if info_res and info_res.returncode == 0 and "nvidia" in info_res.stdout.lower():
        nvidia_runtime = True
        
    return True, docker_version, nvidia_runtime


def ensure_docker_running(timeout_sec: int = 60) -> bool:
    """
    Intelligently verifies if Docker is running.
    If installed but stopped, it automatically launches Docker Desktop in the background
    and waits for the daemon to become ready!
    """
    is_running, ver, _ = check_docker_environment()
    if is_running:
        return True

    print("\n🐳 Docker is not currently responding. Checking for local installation...")

    if sys.platform.startswith("win"):
        # Common Docker Desktop installation locations on Windows
        possible_paths = [
            r"C:\Program Files\Docker\Docker\Docker Desktop.exe",
            r"C:\Program Files (x86)\Docker\Docker\Docker Desktop.exe",
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), r"Docker\Docker\Docker Desktop.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Docker\Docker Desktop.exe"),
        ]

        exe_path = next((p for p in possible_paths if os.path.exists(p)), None)

        if exe_path:
            print(f"  🚀 Found Docker Desktop at: {exe_path}")
            print("  ⏳ Launching Docker Desktop in background...")
            try:
                subprocess.Popen([exe_path], close_fds=True)
            except Exception as e:
                print(f"  ⚠️ Could not launch Docker executable: {e}")

            # Poll until Docker daemon responds
            start_t = time.time()
            while time.time() - start_t < timeout_sec:
                time.sleep(3)
                elapsed = int(time.time() - start_t)
                is_running, ver, _ = check_docker_environment()
                if is_running:
                    print(f"\n  ✅ Docker Desktop is now running and operational (v{ver})!")
                    return True
                print(f"\r  ⏳ Waiting for Docker engine to initialize... ({elapsed}s / {timeout_sec}s)", end="", flush=True)

            print("\n  ❌ Timed out waiting for Docker Desktop to start.")
            return False
        else:
            print("  ❌ Docker Desktop executable not found in standard paths.")
            print("  📥 Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/")
            return False

    elif sys.platform.startswith("linux"):
        print("  ⏳ Attempting to start Docker service via systemctl...")
        run_command(["sudo", "systemctl", "start", "docker"])
        time.sleep(3)
        is_running, ver, _ = check_docker_environment()
        if is_running:
            print(f"  ✅ Docker service started successfully (v{ver})!")
            return True
        print("  ❌ Could not auto-start Docker service. Run: sudo systemctl start docker")
        return False

    return False


def detect_hardware() -> SystemHardwareReport:
    """Execute complete hardware probe with all fallbacks."""
    cpu_model, cpu_cores = get_cpu_info()
    ram_gb = get_system_ram_gb()
    
    # GPU detection pipeline
    backend = "pynvml"
    gpus = detect_gpus_pynvml()
    
    if not gpus:
        backend = "torch"
        gpus = detect_gpus_torch()
        
    if not gpus:
        backend = "nvidia-smi"
        gpus = detect_gpus_nvidiasmi()
        
    if not gpus:
        backend = "none (CPU fallback)"
        gpus = []

    total_vram = round(sum(g.total_vram_gb for g in gpus), 2)
    max_single_vram = round(max((g.total_vram_gb for g in gpus), default=0.0), 2)

    docker_ok, docker_ver, nvidia_runtime = check_docker_environment()

    return SystemHardwareReport(
        os_name=f"{platform.system()} {platform.release()}",
        arch=platform.machine(),
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        system_ram_gb=ram_gb,
        gpus=gpus,
        total_vram_gb=total_vram,
        max_single_gpu_vram_gb=max_single_vram,
        docker_installed=docker_ok,
        docker_version=docker_ver,
        nvidia_runtime_available=nvidia_runtime,
        detection_backend=backend,
    )


def print_hardware_summary(report: SystemHardwareReport):
    """Print a clean visual diagnostic summary to console."""
    sep = "=" * 70
    print(sep)
    print("  🚀 ENTERPRISE HARDWARE PROBE REPORT (Self-Hosted AI Node)")
    print(sep)
    print(f"  OS               : {report.os_name} ({report.arch})")
    print(f"  CPU              : {report.cpu_model} ({report.cpu_cores} Cores)")
    print(f"  System RAM       : {report.system_ram_gb} GB")
    print(f"  Probe Backend    : {report.detection_backend}")
    print("-" * 70)
    
    if report.gpus:
        for g in report.gpus:
            temp_str = f", Temp: {g.temperature_c}°C" if g.temperature_c else ""
            print(f"  🎮 GPU #{g.index:<2} : {g.name:<26} | VRAM: {g.total_vram_gb}GB (Free: {g.free_vram_gb}GB)")
            print(f"               Driver: {g.driver_version}, Compute Cap: {g.compute_capability}{temp_str}")
        print(f"  📊 Total VRAM    : {report.total_vram_gb} GB across {len(report.gpus)} GPU(s)")
        print(f"  ⚡ Max Single GPU: {report.max_single_gpu_vram_gb} GB")
    else:
        print("  ⚠️  GPU          : No NVIDIA GPU detected. System will default to CPU inference.")

    print("-" * 70)
    docker_status = f"✅ Active (v{report.docker_version})" if report.docker_installed else "❌ Not Running / Not Found"
    nvidia_status = "✅ Active" if report.nvidia_runtime_available else "❌ Not Configured (nvidia-container-toolkit required for GPUs)"
    print(f"  🐳 Docker Engine : {docker_status}")
    print(f"  🔌 NVIDIA Runtime: {nvidia_status}")
    print(sep)


if __name__ == "__main__":
    report = detect_hardware()
    if "--json" in sys.argv:
        print(json.dumps(asdict(report), indent=2))
    else:
        print_hardware_summary(report)
