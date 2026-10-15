#!/bin/bash
#SBATCH --job-name=aisci_v1_nanogpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --chdir=/u/qd0037/SWM/AI-Scientist
#SBATCH --output=logs/aisci_v1_nanogpt_%j.out
#SBATCH --error=logs/aisci_v1_nanogpt_%j.err

set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

PROJECT_DIR="${PROJECT_DIR:-/u/qd0037/SWM/AI-Scientist}"
CONDA_ENV="${CONDA_ENV:-/n/fs/vision-mix/qd0037/conda_envs/ai_scientist}"
AI_SCIENTIST_ENV_FILE="${AI_SCIENTIST_ENV_FILE:-$PROJECT_DIR/.env}"
EXPERIMENT="${EXPERIMENT:-nanoGPT}"
MODEL="${MODEL:-codex-headless}"
REVIEW_MODEL="${REVIEW_MODEL:-codex-headless}"
ENGINE="${ENGINE:-semanticscholar}"
NUM_IDEAS="${NUM_IDEAS:-500}"
NOVELTY_CHECK_ROUNDS="${NOVELTY_CHECK_ROUNDS:-1}"
GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
GPU_COUNT="${#GPU_ARRAY[@]}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
PARALLEL="${PARALLEL:-$((GPU_COUNT * WORKERS_PER_GPU))}"
ALLOW_GPU_OVERSUBSCRIPTION="${ALLOW_GPU_OVERSUBSCRIPTION:-auto}"
if [ "$ALLOW_GPU_OVERSUBSCRIPTION" = "auto" ]; then
    if [ "$PARALLEL" -gt "$GPU_COUNT" ]; then
        ALLOW_GPU_OVERSUBSCRIPTION=1
    else
        ALLOW_GPU_OVERSUBSCRIPTION=0
    fi
fi
BASELINE_GPU="${BASELINE_GPU:-0}"
FORCE_BASELINE="${FORCE_BASELINE:-0}"
IMPROVEMENT="${IMPROVEMENT:-1}"
SKIP_IDEA_GENERATION="${SKIP_IDEA_GENERATION:-0}"
USE_EXISTING_NOVELTY="${USE_EXISTING_NOVELTY:-0}"
STOP_AFTER_NOVELTY_CHECK="${STOP_AFTER_NOVELTY_CHECK:-0}"
GPU_KEEPALIVE="${GPU_KEEPALIVE:-1}"
GPU_KEEPALIVE_INTERVAL="${GPU_KEEPALIVE_INTERVAL:-20}"
GPU_KEEPALIVE_MATRIX_SIZE="${GPU_KEEPALIVE_MATRIX_SIZE:-256}"
# Keep novelty filtering on for this run script.
SKIP_NOVELTY_CHECK=0
export CODEX_HEADLESS_MODEL="${CODEX_HEADLESS_MODEL:-gpt-5.5}"
export CODEX_HEADLESS_TIMEOUT="${CODEX_HEADLESS_TIMEOUT:-1800}"
export CODEX_HEADLESS_SANDBOX="${CODEX_HEADLESS_SANDBOX:-read-only}"
export CODEX_HEADLESS_EPHEMERAL="${CODEX_HEADLESS_EPHEMERAL:-1}"
export CODEX_HEADLESS_SKIP_GIT_CHECK="${CODEX_HEADLESS_SKIP_GIT_CHECK:-1}"
export CODEX_CODER_TIMEOUT="${CODEX_CODER_TIMEOUT:-1800}"
export CODEX_CODER_SANDBOX="${CODEX_CODER_SANDBOX:-workspace-write}"
export AI_SCIENTIST_SEARCH_BACKOFF_MAX_TRIES="${AI_SCIENTIST_SEARCH_BACKOFF_MAX_TRIES:-6}"
export AI_SCIENTIST_SEARCH_BACKOFF_MAX_TIME="${AI_SCIENTIST_SEARCH_BACKOFF_MAX_TIME:-180}"
export AI_SCIENTIST_S2_MIN_INTERVAL="${AI_SCIENTIST_S2_MIN_INTERVAL:-4.0}"
export AI_SCIENTIST_NOVELTY_MAX_ROUNDS="$NOVELTY_CHECK_ROUNDS"

cd "$PROJECT_DIR"
mkdir -p logs "results/$EXPERIMENT"

if [ -f "$AI_SCIENTIST_ENV_FILE" ]; then
    set -a
    source "$AI_SCIENTIST_ENV_FILE"
    set +a
fi

echo "=============================="
echo "AI-Scientist v1 nanoGPT pipeline started at: $(date)"
echo "Project dir: $PROJECT_DIR"
echo "Conda env: $CONDA_ENV"
echo "Env file: $AI_SCIENTIST_ENV_FILE"
echo "Experiment: $EXPERIMENT"
echo "Model: $MODEL"
echo "Review model: $REVIEW_MODEL"
echo "Codex model: ${CODEX_HEADLESS_MODEL:-default}"
echo "Engine: $ENGINE"
if [ -n "${S2_API_KEY:-}" ]; then
    echo "Semantic Scholar API key: loaded"
else
    echo "Semantic Scholar API key: not set"
fi
echo "Num ideas: $NUM_IDEAS"
echo "Novelty check rounds: $NOVELTY_CHECK_ROUNDS"
echo "Parallel workers: $PARALLEL"
echo "GPU IDs: $GPUS"
echo "Workers per GPU: $WORKERS_PER_GPU"
echo "Allow GPU oversubscription: $ALLOW_GPU_OVERSUBSCRIPTION"
echo "Improvement enabled: $IMPROVEMENT"
echo "Novelty check enabled: yes"
echo "Use existing novelty labels: $USE_EXISTING_NOVELTY"
echo "Stop after novelty check: $STOP_AFTER_NOVELTY_CHECK"
echo "GPU keepalive enabled: $GPU_KEEPALIVE"
echo "GPU keepalive interval: $GPU_KEEPALIVE_INTERVAL seconds"
echo "Semantic Scholar minimum interval: $AI_SCIENTIST_S2_MIN_INTERVAL seconds"
echo "SLURM job id: ${SLURM_JOB_ID:-manual}"
echo "=============================="

if type module >/dev/null 2>&1; then
    module purge
    module load anaconda3/2024.02 || true
    module load proxy/default >/dev/null 2>&1 || true
fi

if [ -f /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh ]; then
    source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: conda was not found. Load anaconda or set CONDA_ENV to an existing Python env." >&2
    exit 1
fi

conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

resolve_codex_bin() {
    if [ -n "${CODEX_BIN:-}" ]; then
        if [ -x "$CODEX_BIN" ]; then
            export CODEX_BIN
            return 0
        fi
        echo "WARNING: CODEX_BIN is set but not executable: $CODEX_BIN" >&2
    fi

    local found
    found="$(command -v codex 2>/dev/null || true)"
    if [ -n "$found" ]; then
        export CODEX_BIN="$found"
        return 0
    fi

    shopt -s nullglob
    local candidates=(
        "$HOME/.local/bin/codex"
        "$HOME/bin/codex"
        "$HOME/.cursor-server/extensions"/openai.chatgpt-*/bin/linux-x86_64/codex
    )
    shopt -u nullglob

    local candidate
    for candidate in "${candidates[@]}"; do
        if [ -x "$candidate" ]; then
            export CODEX_BIN="$candidate"
            export PATH="$(dirname "$candidate"):$PATH"
            return 0
        fi
    done

    return 1
}

install_codex_dynamic_wrapper() {
    local initial_bin="${CODEX_BIN:-}"
    local wrapper_dir="$PROJECT_DIR/.slurm_libs"
    local wrapper="$wrapper_dir/codex_dynamic"
    mkdir -p "$wrapper_dir"
    cat > "$wrapper" <<'EOF'
#!/bin/bash
set -euo pipefail

best=""

consider_codex() {
    local candidate="$1"
    if [ -x "$candidate" ]; then
        if [ -z "$best" ] || [ "$candidate" -nt "$best" ]; then
            best="$candidate"
        fi
    fi
}

if [ -n "${CODEX_REAL_BIN:-}" ]; then
    consider_codex "$CODEX_REAL_BIN"
fi

found="$(command -v codex 2>/dev/null || true)"
if [ -n "$found" ] && [ "$found" != "$0" ]; then
    consider_codex "$found"
fi

shopt -s nullglob
for candidate in \
    "$HOME/.local/bin/codex" \
    "$HOME/bin/codex" \
    "$HOME/.cursor-server/extensions"/openai.chatgpt-*/bin/linux-x86_64/codex
do
    consider_codex "$candidate"
done
shopt -u nullglob

if [ -z "$best" ]; then
    echo "ERROR: No codex executable found. Set CODEX_BIN or install/open the Codex extension." >&2
    exit 127
fi

exec "$best" "$@"
EOF
    chmod +x "$wrapper"
    export CODEX_REAL_BIN="$initial_bin"
    export CODEX_BIN="$wrapper"
}

resolve_codex_bin || true
if [ -n "${CODEX_BIN:-}" ]; then
    install_codex_dynamic_wrapper
    echo "Codex executable wrapper: $CODEX_BIN"
    CODEX_VERSION="$("$CODEX_BIN" --version 2>/dev/null || echo unknown)"
    echo "Codex version: $CODEX_VERSION"
fi

# Work around PyTorch builds that expect Intel ITT symbols from libittnotify.
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

require_key_for_model() {
    local model_name="$1"
    case "$model_name" in
        codex-headless*)
            command -v "${CODEX_BIN:-codex}" >/dev/null 2>&1 || {
                echo "ERROR: codex-headless requested but no codex executable was found. Set CODEX_BIN or add codex to PATH." >&2
                exit 1
            }
            ;;
        claude-*)
            [ -n "${ANTHROPIC_API_KEY:-}" ] || {
                echo "ERROR: ANTHROPIC_API_KEY is required for MODEL=$model_name" >&2
                exit 1
            }
            ;;
        gpt*|o1*|o3*)
            [ -n "${OPENAI_API_KEY:-}" ] || {
                echo "ERROR: OPENAI_API_KEY is required for model $model_name" >&2
                exit 1
            }
            ;;
        deepseek-*)
            [ -n "${DEEPSEEK_API_KEY:-}" ] || {
                echo "ERROR: DEEPSEEK_API_KEY is required for MODEL=$model_name" >&2
                exit 1
            }
            ;;
        llama3.1-405b)
            [ -n "${OPENROUTER_API_KEY:-}" ] || {
                echo "ERROR: OPENROUTER_API_KEY is required for MODEL=$model_name" >&2
                exit 1
            }
            ;;
        gemini*)
            [ -n "${GEMINI_API_KEY:-}" ] || {
                echo "ERROR: GEMINI_API_KEY is required for MODEL=$model_name" >&2
                exit 1
            }
            ;;
    esac
}

require_key_for_model "$MODEL"
require_key_for_model "$REVIEW_MODEL"

GPU_KEEPALIVE_PIDS=()

stop_gpu_keepalive() {
    if [ "${#GPU_KEEPALIVE_PIDS[@]}" -gt 0 ]; then
        echo "Stopping GPU keepalive processes..."
        kill "${GPU_KEEPALIVE_PIDS[@]}" >/dev/null 2>&1 || true
        wait "${GPU_KEEPALIVE_PIDS[@]}" >/dev/null 2>&1 || true
        GPU_KEEPALIVE_PIDS=()
    fi
}

trap stop_gpu_keepalive EXIT
trap 'stop_gpu_keepalive; exit 130' INT
trap 'stop_gpu_keepalive; exit 143' TERM

start_gpu_keepalive() {
    if [ "$GPU_KEEPALIVE" != "1" ]; then
        echo "GPU keepalive disabled."
        return 0
    fi

    echo "Starting GPU keepalive on GPU IDs: $GPUS"
    local gpu_id
    for gpu_id in "${GPU_ARRAY[@]}"; do
        (
            export CUDA_VISIBLE_DEVICES="$gpu_id"
            export KEEPALIVE_GPU_ID="$gpu_id"
            export GPU_KEEPALIVE_INTERVAL
            export GPU_KEEPALIVE_MATRIX_SIZE
            python -u - <<'PY'
import os
import time

import torch

gpu_id = os.environ.get("KEEPALIVE_GPU_ID", "0")
interval = float(os.environ.get("GPU_KEEPALIVE_INTERVAL", "20"))
size = int(os.environ.get("GPU_KEEPALIVE_MATRIX_SIZE", "256"))

if not torch.cuda.is_available():
    print(f"[gpu-keepalive:{gpu_id}] CUDA is not available; exiting.", flush=True)
    raise SystemExit(0)

device = torch.device("cuda:0")
a = torch.randn((size, size), device=device)
b = torch.randn((size, size), device=device)
print(
    f"[gpu-keepalive:{gpu_id}] running tiny matmul every {interval:g}s "
    f"with matrix size {size}.",
    flush=True,
)

while True:
    c = (a @ b) / size
    torch.cuda.synchronize(device)
    # Keep the tensors changing so each loop launches real CUDA work.
    a = torch.tanh(c)
    time.sleep(interval)
PY
        ) &
        GPU_KEEPALIVE_PIDS+=("$!")
    done
}

echo "=============================="
echo "Environment"
echo "Python: $(which python)"
echo "Python version: $(python -V)"
echo "pdflatex: $(command -v pdflatex || true)"
echo "chktex: $(command -v chktex || true)"
echo "latexmk: $(command -v latexmk || true)"
nvidia-smi || true
echo "=============================="

EXPECTED_GPU_COUNT="$GPU_COUNT" python - <<'PY'
import os
import shutil
import sys

missing = [name for name in ("pdflatex", "chktex") if shutil.which(name) is None]
if missing:
    print(
        "ERROR: missing LaTeX tools: "
        + ", ".join(missing)
        + "\nRun ./setup_ai_scientist_v1_env.sh first. It installs/checks these via conda, not sudo apt.",
        file=sys.stderr,
    )
    sys.exit(1)

import torch
expected_gpu_count = int(os.environ.get("EXPECTED_GPU_COUNT", "1"))
visible_gpu_count = torch.cuda.device_count()
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "gpu_count:", visible_gpu_count)
if visible_gpu_count < expected_gpu_count:
    print(
        f"WARNING: expected {expected_gpu_count} CUDA device(s) from GPUS, but only {visible_gpu_count} are visible to the job.",
        file=sys.stderr,
    )
PY

if [ "$EXPERIMENT" = "nanoGPT" ] || [ "$EXPERIMENT" = "nanoGPT_lite" ]; then
    if [ ! -s data/shakespeare_char/train.bin ] || [ ! -s data/shakespeare_char/val.bin ]; then
        echo "Preparing Shakespeare character data..."
        python data/shakespeare_char/prepare.py
    fi
fi

BASELINE_DIR="templates/$EXPERIMENT/run_0"
if [ "$FORCE_BASELINE" = "1" ] || [ ! -s "$BASELINE_DIR/final_info.json" ] || [ ! -s "$BASELINE_DIR/all_results.npy" ]; then
    echo "Creating baseline run for $EXPERIMENT on CUDA_VISIBLE_DEVICES=$BASELINE_GPU..."
    mkdir -p "$BASELINE_DIR"
    (
        cd "templates/$EXPERIMENT"
        export CUDA_VISIBLE_DEVICES="$BASELINE_GPU"
        python -u experiment.py --out_dir run_0
        python -u plot.py
    )
else
    echo "Baseline exists at $BASELINE_DIR; set FORCE_BASELINE=1 to rebuild it."
fi

start_gpu_keepalive

CMD=(
    python -u launch_scientist.py
    --model "$MODEL"
    --review-model "$REVIEW_MODEL"
    --experiment "$EXPERIMENT"
    --num-ideas "$NUM_IDEAS"
    --novelty-check-rounds "$NOVELTY_CHECK_ROUNDS"
    --parallel "$PARALLEL"
    --gpus "$GPUS"
    --engine "$ENGINE"
)

if [ "$IMPROVEMENT" = "1" ]; then
    CMD+=(--improvement)
fi
if [ "$ALLOW_GPU_OVERSUBSCRIPTION" = "1" ]; then
    CMD+=(--allow-gpu-oversubscription)
fi
if [ "$SKIP_IDEA_GENERATION" = "1" ]; then
    CMD+=(--skip-idea-generation)
fi
if [ "$USE_EXISTING_NOVELTY" = "1" ]; then
    CMD+=(--use-existing-novelty)
fi
if [ "$STOP_AFTER_NOVELTY_CHECK" = "1" ]; then
    CMD+=(--stop-after-novelty-check)
fi
if [ "$SKIP_NOVELTY_CHECK" = "1" ]; then
    CMD+=(--skip-novelty-check)
fi

echo "Launching:"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "AI-Scientist v1 pipeline finished at: $(date)"
