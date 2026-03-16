#!/bin/bash
#
# Small Slurm job to render GRF episodes to video (for qualitative visualization).
# Usage (from repo root):
#   sbatch scripts/slurm_visualize.sh outputs/5089899 configs/llm/reward/3v1_config.yaml
# The first argument is the run output_dir (must contain agents/ or checkpoints/),
# the second is the training config used for that run.
#

# Job settings — adjust partition/time if needed by your cluster policy
# For CS5180, courses-gpu is usually available; if not, change to an allowed GPU partition.
#SBATCH --job-name=grf-vis
#SBATCH --partition=courses-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
# Logs go under the analysis / run directory
#SBATCH --output=outputs/vis_%j.out
#SBATCH --error=outputs/vis_%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch scripts/slurm_visualize.sh <output_dir> [config_path] [episodes]" >&2
  echo "  <output_dir>: run directory, e.g. outputs/5089899" >&2
  echo "  [config_path]: training config used for that run (default: configs/llm/reward/3v1_config.yaml)" >&2
  echo "  [episodes]: number of episodes to record (default: 3)" >&2
  exit 1
fi

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

OUTPUT_DIR_REL="$1"
CONFIG_PATH_REL="${2:-configs/llm/reward/3v1_config.yaml}"
EPISODES="${3:-3}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR_REL}"
echo "CONFIG=${CONFIG_PATH_REL}"
echo "EPISODES=${EPISODES}"

# Activate training environment (same as other jobs)
if [[ -f "${REPO_ROOT}/environment.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/environment.sh"
fi

python3 scripts/visualize.py \
  --config "${CONFIG_PATH_REL}" \
  --output-dir "${OUTPUT_DIR_REL}" \
  --episodes "${EPISODES}"

echo "Visualization job finished."

