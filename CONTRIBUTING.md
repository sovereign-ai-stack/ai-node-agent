# Contributing to Sovereign AI Node Agent

Thank you for your interest in contributing to the Sovereign AI Node Agent!

The Node Agent powers autonomous, high-throughput model execution across distributed, heterogenous consumer GPUs.

## Development Workflow

1. Fork the repo and clone locally:
   ```bash
   git clone https://github.com/your-username/ai-node-agent.git
   cd ai-node-agent
   ```
2. Install development tools:
   ```bash
   pip install -r requirements.txt
   pip install pytest ruff
   ```
3. Test hardware discovery locally:
   ```bash
   python -c "from hardware_detector import detect_hardware, print_hardware_summary; hw = detect_hardware(); print_hardware_summary(hw)"
   ```

## Pull Request Guidelines

- Branch naming: `feature/your-feature-name` or `fix/your-fix-name`.
- Use descriptive commit messages following Conventional Commits format.
- Ensure any modifications to `hardware_detector.py` maintain fallback compatibility across Linux, WSL2, and native Windows.
