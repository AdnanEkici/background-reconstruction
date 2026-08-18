#!/usr/bin/env bash
set -Eeuo pipefail

: <<'DOC'
===============================================================================
Virtual Environment Setup Script
===============================================================================

This script creates a Python virtual environment (`.venv`) in the project root,
upgrades `pip`, and installs all required dependencies for development,
including machine learning and image processing.

-------------------------------------------------------------------------------
📦 Installed Packages
-------------------------------------------------------------------------------
- numpy==2.3.1                → Numerical computing
- opencv-python==4.11.0.86    → Computer vision (OpenCV)
- setuptools==80.9.0          → Packaging/build utilities (pinned)
- torch (CUDA 12.9 build)     → Installed via PyTorch wheel index
- torchvision                 → Vision utilities for PyTorch
- pyyaml==6.0.2               → YAML parsing
- tqdm==4.67.1                → Progress bars
- pre-commit==4.1.0           → Git hooks
- lpips==0.1.4                → Perceptual similarity metric
- sphinx==8.2.3               → Documentation generator
- sphinx-rtd-theme==3.0.2     → ReadTheDocs theme

⚠️ Note: torch and torchvision are installed from the official PyTorch wheel index:
       https://download.pytorch.org/whl/cu129

-------------------------------------------------------------------------------
▶️ Usage
-------------------------------------------------------------------------------
1. Run the script:
       ./scripts/setup_venv.sh

2. When prompted, confirm with "yes".

3. Activate the virtual environment:
       source .venv/bin/activate

4. Deactivate with:
       deactivate
===============================================================================
DOC

# ================= Colors =================
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
BLUE='\033[0;34m'; PURPLE='\033[0;35m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'

# ================= Settings ================
VENV_PATH=".venv"

# ================= Helpers =================
banner() { echo -e "${BOLD}${CYAN}==> $*${RESET}"; }
info()   { echo -e "${BLUE}•${RESET} $*"; }
warn()   { echo -e "${YELLOW}!${RESET} $*"; }
ok()     { echo -e "${GREEN}✔${RESET} $*"; }
fail()   { echo -e "${RED}✖ ${BOLD}$*${RESET}"; exit 1; }

confirm_action() {
    while true; do
        read -r -p "$(echo -e "${PURPLE}Create a virtual environment in '$VENV_PATH'? (yes/no): ${RESET}")" choice
        case "${choice,,}" in
            y|yes) return 0 ;;
            n|no)  return 1 ;;
            *)     warn "Please answer 'yes' or 'no'." ;;
        esac
    done
}

# ================= Script =================
banner "Virtual Environment Setup"

if confirm_action; then
    command -v python3 >/dev/null 2>&1 || fail "python3 is not installed."

    banner "Creating virtual environment at ${VENV_PATH}"
    python3 -m venv "$VENV_PATH"

    pip_bin="$VENV_PATH/bin/pip"

    banner "Upgrading pip"
    "$pip_bin" install --upgrade pip || fail "pip upgrade failed."

    banner "Installing base packages"
    "$pip_bin" install \
        numpy==2.3.1 \
        opencv-python==4.11.0.86 \
        setuptools==80.9.0 \
        pyyaml==6.0.2 \
        tqdm==4.67.1 \
        pre-commit==4.1.0 \
        lpips==0.1.4 \
        sphinx==8.2.3 \
        sphinx-rtd-theme==3.0.2 || fail "base packages failed."

    banner "Installing PyTorch (CUDA 12.9)"
    "$pip_bin" install torch torchvision --index-url https://download.pytorch.org/whl/cu129 || fail "torch install failed."

    ok "Virtual environment setup complete."
    info "Activate it with:"
    info "   source $VENV_PATH/bin/activate"
else
    echo -e "${BLUE}Aborted. No action taken.${RESET}"
fi
