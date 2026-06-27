#!/usr/bin/env bash
# Run this ONCE inside the Azure ML compute instance `lwmf-t4`
# (AML Studio -> Compute -> lwmf-t4 -> Terminal). It clones the repo,
# builds a venv with the GPU deps, and verifies CUDA + the offline suite.
#
# Private-repo auth: before the clone, authenticate git on the compute, e.g.
#   gh auth login        # if gh is available, OR
#   git config --global credential.helper store  # then clone with a PAT
set -euo pipefail

REPO_URL="https://github.com/JaviMaligno/language-world-model-forgetting.git"
cd ~
if [ ! -d language-world-model-forgetting ]; then
  git clone "$REPO_URL"
fi
cd language-world-model-forgetting
git pull --ff-only || true

# Fresh venv (compute instance already has NVIDIA driver + CUDA runtime)
python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev,gpu]"

echo "=== CUDA check ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'))"

echo "=== offline unit suite ==="
pytest -q -m "not smoke and not gpu"

echo "=== GPU smoke (Task 13: real-tokenizer mask + overfit-one-batch) ==="
pytest -q -m smoke || echo "(smoke needs network for tokenizer download + GPU; rerun if it failed transiently)"

echo "Bootstrap done. Next: scripts/RUNBOOK_AML.md (Phase 0)."
