#!/usr/bin/env bash
# ------------------------------------------------------------------
# Setup a dedicated Python 3.11 virtualenv for ACE-Step 1.5
#
# Needed because ACE-Step requires:
#   - Python 3.11 (our main env is 3.12)
#   - transformers >= 4.57 (MiniCPM-o needs == 4.51.0)
#
# Usage:
#   bash setup_ace_step.sh
# ------------------------------------------------------------------

set -euo pipefail

VENV_NAME="ace-step"
PY_VERSION="3.11.12"
REPO_DIR="/home/ubuntu/github/ACE-Step-1.5"
MODEL_DIR="/home/ubuntu/karthik-ragunath-ananda-kumar-utah/Ace-Step1.5"

echo "=== ACE-Step 1.5 Environment Setup ==="

# 1. Install Python 3.11 if not present
eval "$(pyenv init -)"
if ! pyenv versions --bare | grep -q "^${PY_VERSION}$"; then
    echo "Installing Python ${PY_VERSION} via pyenv ..."
    pyenv install "${PY_VERSION}"
fi

# 2. Create virtualenv
if pyenv virtualenvs --bare | grep -q "^${VENV_NAME}$"; then
    echo "Virtualenv '${VENV_NAME}' already exists, skipping creation."
else
    echo "Creating virtualenv '${VENV_NAME}' ..."
    pyenv virtualenv "${PY_VERSION}" "${VENV_NAME}"
fi

# 3. Activate and install
pyenv activate "${VENV_NAME}"
echo "Python: $(python --version)"

echo "Installing ACE-Step 1.5 from local repo ..."
pip install --upgrade pip
pip install -e "${REPO_DIR}"

echo ""
echo "=== Setup complete ==="
echo "Model weights:  ${MODEL_DIR}"
echo "Repo:           ${REPO_DIR}"
echo "Virtualenv:     ${VENV_NAME}"
echo ""
echo "To use:  pyenv activate ${VENV_NAME}"
echo "         python /home/ubuntu/github/minicpm-o-4_5/inference_ace_step.py"
