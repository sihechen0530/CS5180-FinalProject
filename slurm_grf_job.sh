#!/bin/bash
#
# Usage:
#   sbatch slurm_grf_job.sh train configs/empty_config.yaml [output_dir]
#   sbatch slurm_grf_job.sh eval  configs/empty_config.yaml <output_dir> [--agent path/to/ppo.zip]
#
# Train: If output_dir is omitted, outputs/<SLURM_JOB_ID> is used. All run
# artifacts and Slurm logs go under that directory.
#
# Eval: output_dir (3rd arg) is required — you must point to an existing run
# (e.g. outputs/12345 or outputs/my_run). Evaluation log and Slurm job log
# are both written under that directory (logs/). No separate folder is created.
#
#SBATCH --job-name=grf
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
# Slurm stdout/stderr go under the job output dir (outputs/%j/logs/ when using default)
#SBATCH --output=outputs/%j/logs/slurm-%x-%j.out
#SBATCH --error=outputs/%j/logs/slurm-%x-%j.err

set -euo pipefail

# -------------------------- Repo & arguments ---------------------------------
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="${SCRIPT_DIR}"
fi

MODE="${1:-train}"
CONFIG_REL="${2:-configs/empty_config.yaml}"
OUTPUT_DIR="${3:-}"
AGENT_OVERRIDE="${4:-}"

cd "${REPO_ROOT}"

# Eval requires an existing run directory; train can default to outputs/<job_id>
if [[ "${MODE}" == "eval" ]]; then
  if [[ -z "${OUTPUT_DIR}" ]]; then
    echo "For eval mode, output_dir (3rd argument) is required. Point to an existing run directory." >&2
    exit 1
  fi
else
  if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="outputs/${SLURM_JOB_ID}"
  fi
fi

if [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${REPO_ROOT}/${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/videos"

LOG_DIR="${OUTPUT_DIR}/logs"
TRAIN_LOG="${LOG_DIR}/train_${SLURM_JOB_ID}.log"
EVAL_LOG="${LOG_DIR}/eval_${SLURM_JOB_ID}.log"

echo "REPO_ROOT=${REPO_ROOT}"
echo "MODE=${MODE}"
echo "CONFIG=${CONFIG_REL}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

if [[ -n "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is detected and will be forwarded to the environment."
  export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}"
else
  echo "No DEEPSEEK_API_KEY detected in the environment."
fi

# -------------------------- Conda & modules ----------------------------------
source "${REPO_ROOT}/environment.sh"

# -------------------------- Run mode ----------------------------------------

if [[ "${MODE}" == "train" ]]; then
  echo "Starting training; log -> ${TRAIN_LOG}"
  python3 scripts/train.py --config "${CONFIG_REL}" --output-dir "${OUTPUT_DIR}" > "${TRAIN_LOG}" 2>&1

elif [[ "${MODE}" == "eval" ]]; then
  echo "Starting evaluation; log -> ${EVAL_LOG}"
  CMD=(python3 scripts/evaluate.py --config "${CONFIG_REL}" --output-dir "${OUTPUT_DIR}")
  if [[ -n "${AGENT_OVERRIDE}" ]]; then
    CMD+=(--agent "${AGENT_OVERRIDE}")
  fi
  echo "Command: ${CMD[*]}"
  "${CMD[@]}" > "${EVAL_LOG}" 2>&1

else
  echo "Unknown MODE='${MODE}'. Use 'train' or 'eval'." >&2
  exit 1
fi

echo "Job ${SLURM_JOB_ID} finished. Output directory: ${OUTPUT_DIR}"
