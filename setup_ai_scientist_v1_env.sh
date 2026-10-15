#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/u/qd0037/SWM/AI-Scientist}"
CONDA_ENV="${CONDA_ENV:-/n/fs/vision-mix/qd0037/conda_envs/ai_scientist}"
INSTALL_OPENALEX="${INSTALL_OPENALEX:-0}"

cd "$PROJECT_DIR"

if type module >/dev/null 2>&1; then
    module load anaconda3/2024.02 || true
    module load proxy/default >/dev/null 2>&1 || true
fi

if [ -f /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh ]; then
    source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: conda was not found." >&2
    exit 1
fi

if ! conda env list | awk '{print $NF}' | grep -Fxq "$CONDA_ENV"; then
    echo "Creating conda env at $CONDA_ENV"
    conda create -y -p "$CONDA_ENV" python=3.11
fi

conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m pip install --upgrade pip
# Avoid installing the repo's unpinned requirements file directly. In 2026,
# pip can backtrack to very old aider/openai versions that break openai.OpenAI().
python -m pip install --upgrade \
    "aider-chat==0.86.2" \
    "openai>=1.99,<3" \
    "anthropic>=0.34" \
    "numpy>=1.26.4" \
    "packaging>=24" \
    "pathspec>=1" \
    backoff \
    google-generativeai \
    matplotlib \
    pypdf \
    pymupdf4llm \
    torch \
    transformers \
    "datasets>=4.8.5" \
    tiktoken \
    wandb \
    tqdm

if ! command -v chktex >/dev/null 2>&1; then
    conda install -y -c conda-forge chktex
fi

if ! command -v pdflatex >/dev/null 2>&1; then
    echo "pdflatex is missing; installing texlive-core via conda-forge."
    conda install -y -c conda-forge texlive-core
fi

if [ "$INSTALL_OPENALEX" = "1" ]; then
    python -m pip install pyalex
fi

ITT_STUB_DIR="$PROJECT_DIR/.slurm_libs"
ITT_STUB="$ITT_STUB_DIR/libittnotify_stub.so"
if [ ! -f "$ITT_STUB" ]; then
    mkdir -p "$ITT_STUB_DIR"
    cat > "$ITT_STUB_DIR/ittnotify_stub.c" <<'EOF'
int iJIT_NotifyEvent(unsigned int event_type, void *event_specific_data) {
    (void)event_type;
    (void)event_specific_data;
    return 0;
}

int iJIT_IsProfilingActive(void) {
    return 0;
}

unsigned int iJIT_GetNewMethodID(void) {
    static unsigned int method_id = 1;
    return method_id++;
}
EOF
    gcc -shared -fPIC "$ITT_STUB_DIR/ittnotify_stub.c" -o "$ITT_STUB"
fi
export LD_PRELOAD="$ITT_STUB${LD_PRELOAD:+:$LD_PRELOAD}"

echo "=============================="
echo "Environment check"
echo "Python: $(which python)"
python -V
echo "pdflatex: $(command -v pdflatex || true)"
echo "chktex: $(command -v chktex || true)"
echo "latexmk: $(command -v latexmk || true)"
echo "codex: $(command -v "${CODEX_BIN:-codex}" || true)"
if ! command -v "${CODEX_BIN:-codex}" >/dev/null 2>&1; then
    echo "WARNING: codex executable not found. MODEL=codex-headless will fail until Codex is on PATH or CODEX_BIN is set." >&2
fi
python - <<'PY'
import importlib
import openai
from aider.coders import Coder
from aider.models import Model

for name in ["torch", "openai", "aider", "pypdf", "pymupdf4llm", "matplotlib", "tiktoken"]:
    importlib.import_module(name)
    print(f"{name}: ok")
print("openai.OpenAI:", hasattr(openai, "OpenAI"))
print("aider classes:", Coder.__name__, Model.__name__)
PY
echo "=============================="
echo "Setup check complete. Submit with:"
echo "sbatch run_ai_scientist_v1_nanogpt_lite.slurm"
