#!/usr/bin/env python3
"""
Auto-Registration & Heartbeat Client for Distributed AI Nodes.
Handles:
1. Health verification of the local vLLM instance (/health and /v1/models).
2. Dynamic registration to the Central Gateway / Registry.
3. Periodic background Heartbeat reporting node status & live load.
4. Graceful de-registration on termination.
"""

import json
import logging
import signal
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [AutoRegister] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("AutoRegister")


def http_request(
    method: str,
    url: str,
    body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 15,
) -> Tuple[int, Dict[str, Any]]:
    """Execute raw HTTP JSON request without third-party dependencies."""
    hdrs = headers or {}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            res_json = json.loads(raw) if raw else {}
            return resp.status, res_json
    except urllib.error.HTTPError as e:
        raw_err = e.read().decode("utf-8", "replace")
        try:
            err_json = json.loads(raw_err)
        except Exception:
            err_json = {"raw": raw_err}
        return e.code, err_json
    except Exception as e:
        return 0, {"error": str(e)}


def get_local_ip(gateway_url: Optional[str] = None) -> str:
    """
    Intelligently detects the exact non-loopback IP address of this machine
    that routes directly to the target gateway (Tailscale 100.x.y.z or Local LAN 192.168.x.x).
    """
    target_host = "10.255.255.255"
    target_port = 1

    if gateway_url:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(gateway_url)
            if parsed.hostname:
                target_host = parsed.hostname
                target_port = parsed.port or 80
        except Exception:
            pass

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((target_host, target_port))
        ip = s.getsockname()[0]
    except Exception:
        # Fallback to local routing check
        try:
            s2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s2.connect(("8.8.8.8", 80))
            ip = s2.getsockname()[0]
            s2.close()
        except Exception:
            ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def wait_for_vllm_ready(api_base: str, timeout_sec: int = 300, interval_sec: int = 5) -> bool:
    """Polls vLLM health endpoint until model weights are loaded and ready."""
    clean_base = api_base.rstrip("/")
    health_url = f"{clean_base}/health"
    models_url = f"{clean_base}/v1/models"
    
    logger.info(f"Checking readiness at {health_url} (timeout: {timeout_sec}s)...")
    deadline = time.time() + timeout_sec
    
    while time.time() < deadline:
        status, data = http_request("GET", health_url, timeout=5)
        if status == 200:
            # Also check if /v1/models returns loaded models
            m_status, m_data = http_request("GET", models_url, timeout=5)
            if m_status == 200 and m_data.get("data"):
                loaded = [m.get("id") for m in m_data.get("data", [])]
                logger.info(f"✅ vLLM is healthy and serving models: {loaded}")
                return True
        logger.info(f"⏳ Waiting for vLLM to finish loading model weights... (retrying in {interval_sec}s)")
        time.sleep(interval_sec)
    
    logger.error("❌ Timeout: vLLM did not become healthy within the allowed window.")
    return False


class NodeRegistrationAgent:
    def __init__(
        self,
        node_id: str,
        gateway_url: str,
        api_base: str,
        model_name: str,
        served_model_name: str,
        hardware_meta: Dict[str, Any],
        supported_roles: Optional[List[str]] = None,
        heartbeat_interval: int = 15,
        auth_token: Optional[str] = None,
    ):
        self.node_id = node_id
        self.gateway_url = gateway_url.rstrip("/")
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.served_model_name = served_model_name
        self.hardware_meta = hardware_meta
        self.supported_roles = supported_roles or []
        self.heartbeat_interval = heartbeat_interval
        self.auth_token = auth_token
        self._stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def register(self) -> bool:
        """Register the node with the central gateway/registry."""
        url = f"{self.gateway_url}/nodes/register"
        payload = {
            "node_id": self.node_id,
            "api_base": self.api_base,
            "model_name": self.model_name,
            "served_model_name": self.served_model_name,
            "supported_roles": self.supported_roles,
            "hardware": self.hardware_meta,
            "timestamp": time.time(),
        }
        
        logger.info(f"Sending registration to {url} ...")
        status, resp = http_request("POST", url, body=payload, headers=self._get_headers())
        
        if status in (200, 201):
            logger.info(f"✅ Successfully registered node '{self.node_id}' with Central Gateway!")
            return True
        
        logger.error(f"❌ Registration failed (HTTP {status}): {resp}")
        return False

    def deregister(self) -> bool:
        """Deregister the node gracefully when shutting down."""
        url = f"{self.gateway_url}/nodes/deregister"
        payload = {"node_id": self.node_id}
        logger.info(f"Deregistering node '{self.node_id}' from {url} ...")
        status, _ = http_request("POST", url, body=payload, headers=self._get_headers())
        return status in (200, 204)

    def start_heartbeat_daemon(self):
        """Starts background periodic heartbeat thread."""
        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(target=self._run_heartbeat, daemon=True)
        self._heartbeat_thread.start()
        logger.info(f"💓 Heartbeat daemon started (interval: {self.heartbeat_interval}s)")

    def stop_heartbeat_daemon(self):
        """Stops background heartbeat thread."""
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._stop_event.set()
            self._heartbeat_thread.join(timeout=5)
            logger.info("Heartbeat daemon stopped.")

    def _run_heartbeat(self):
        url = f"{self.gateway_url}/nodes/heartbeat"
        while not self._stop_event.is_set():
            payload = {
                "node_id": self.node_id,
                "status": "healthy",
                "timestamp": time.time(),
            }
            try:
                status, resp = http_request("POST", url, body=payload, headers=self._get_headers())
                if status not in (200, 204):
                    logger.warning(f"⚠️ Heartbeat returned status {status}: {resp}")
            except Exception as e:
                logger.warning(f"⚠️ Heartbeat error: {e}")

            # Sleep in small increments to be responsive to stop_event
            for _ in range(self.heartbeat_interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)
