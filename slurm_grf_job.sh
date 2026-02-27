#!/bin/bash
#
# Usage:
#   sbatch slurm_grf_job.sh train configs/empty_config.yaml
#   sbatch slurm_grf_job.sh eval  configs/empty_config.yaml
#   sbatch slurm_grf_job.sh eval  configs/3v1_config.yaml agents/baselines/ppo_3v1 outputs/videos_3v1
#
#SBATCH --job-name=grf
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
# Slurm stdout/stderr: go under outputs/ as well
#SBATCH --output=outputs/slurm-%x-%j.out
#SBATCH --error=outputs/slurm-%x-%j.err

set -euo pipefail

# -------------------------- Repo & arguments ---------------------------------
# Use the directory from which sbatch was run (project root). When the job runs,
# the script may execute from a spool dir (e.g. /var/spool/slurmd/jobNNN) so
# BASH_SOURCE would point there and relative paths like configs/ would break.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="${SCRIPT_DIR}"
fi

MODE="${1:-train}"              # train | eval
CONFIG_REL="${2:-configs/empty_config.yaml}"
AGENT_PATH="${3:-}"             # only used in eval mode (optional)
EVAL_LOGDIR_OVERRIDE="${4:-}"   # optional --logdir override for eval

cd "${REPO_ROOT}"
echo "REPO_ROOT=${REPO_ROOT}"

# -------------------------- Conda & modules ----------------------------------
# Adjust these two lines to match your cluster setup

# Ensure conda is available in this non-interactive shell
# e.g. source ~/.bashrc OR the specific conda.sh for your installation
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate /home/chen.sihe1/.conda/envs/grf_env
set -u

# Load FFmpeg module for video conversion
module load FFmpeg

# -------------------------- Resolve output_dir from config -------------------
# This uses the same YAML as your Python scripts to find output_dir,
# then makes a logs/ subdirectory inside it.

OUTPUT_DIR=$(python3 - << 'PY' "${CONFIG_REL}" "${REPO_ROOT}"
import os, sys, yaml

config_rel, repo_root = sys.argv[1], sys.argv[2]
config_path = config_rel
if not os.path.isabs(config_path):
    config_path = os.path.join(repo_root, config_path)

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f) or {}

output_dir = cfg.get("output_dir", "outputs")
if not os.path.isabs(output_dir):
    output_dir = os.path.join(repo_root, output_dir)

print(output_dir)
PY
)

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TRAIN_LOG="${LOG_DIR}/train_${SLURM_JOB_ID}.log"
EVAL_LOG="${LOG_DIR}/eval_${SLURM_JOB_ID}.log"

echo "MODE=${MODE}"
echo "CONFIG=${CONFIG_REL}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_DIR=${LOG_DIR}"

# -------------------------- Run mode ----------------------------------------

if [[ "${MODE}" == "train" ]]; then
  echo "Starting training; log -> ${TRAIN_LOG}"
  python3 scripts/train.py --config "${CONFIG_REL}" > "${TRAIN_LOG}" 2>&1

elif [[ "${MODE}" == "eval" ]]; then
  echo "Starting evaluation; log -> ${EVAL_LOG}"
  CMD=(python3 scripts/evaluate.py --config "${CONFIG_REL}")

  if [[ -n "${AGENT_PATH}" ]]; then
    CMD+=(--agent "${AGENT_PATH}")
  fi

  if [[ -n "${EVAL_LOGDIR_OVERRIDE}" ]]; then
    CMD+=(--logdir "${EVAL_LOGDIR_OVERRIDE}")
  fi

  echo "Command: ${CMD[*]}"
  "${CMD[@]}" > "${EVAL_LOG}" 2>&1

else
  echo "Unknown MODE='${MODE}'. Use 'train' or 'eval'." >&2
  exit 1
fi

echo "Job ${SLURM_JOB_ID} finished."
