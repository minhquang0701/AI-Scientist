#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/u/qd0037/SWM/AI-Scientist}"
cd "$PROJECT_DIR"

IDEAS_SCRIPT="${IDEAS_SCRIPT:-run_ai_scientist_v1_nanogpt_lite_ideas.slurm}"
GPU_SCRIPT="${GPU_SCRIPT:-run_ai_scientist_v1_nanogpt_lite_gpu.slurm}"

ideas_job="$(sbatch --parsable "$IDEAS_SCRIPT")"
gpu_job="$(sbatch --parsable --dependency=afterok:"$ideas_job" "$GPU_SCRIPT")"

echo "Submitted idea + novelty stage: $ideas_job"
echo "Submitted dependent 4-GPU experiment stage: $gpu_job"
echo
echo "Watch:"
echo "  squeue -j $ideas_job,$gpu_job"
echo "  tail -f logs/aisci_v1_ideas_${ideas_job}.out"
echo "  tail -f logs/aisci_v1_gpu_${gpu_job}.out"
