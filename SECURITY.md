# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

Please report sensitive security issues regarding node communication, credential exposure, or container privilege escalation directly to `mahdijm.bb@gmail.com`. Do not submit security reports through public GitHub issues.

## Worker Node Hardening

- The Node Agent should always be isolated inside a Tailscale/WireGuard encrypted overlay.
- Never bind vLLM ports directly to `0.0.0.0` on a public interface; bind strictly to the Tailscale interface (`100.x.y.z`).
