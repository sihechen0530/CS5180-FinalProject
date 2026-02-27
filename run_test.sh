#!/bin/bash
# run_test.sh
#
# Quick sanity check for the empty_config pipeline:
# - Runs training and evaluation
# - Writes outputs/logs to a dedicated test output directory
# - Fails if training times out (>15 min) or eval reward != 1.0

set -euo pipefail

# ----------------------------------------------------------------------
# Configurable parameters
# ----------------------------------------------------------------------

# Config to use
CONFIG="configs/empty_config.yaml"

# Designated test output directory (can be overridden by first arg)
TEST_OUTPUT_DIR="${1:-outputs/run_test}"

# Timeout for training in seconds (15 minutes)
TRAIN_TIMEOUT=900

# ----------------------------------------------------------------------
# Paths and setup
# ----------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${TEST_OUTPUT_DIR}/logs"
VIDEO_DIR="${TEST_OUTPUT_DIR}/videos"
mkdir -p "${LOG_DIR}" "${VIDEO_DIR}"

TRAIN_LOG="${LOG_DIR}/train.log"
EVAL_LOG="${LOG_DIR}/eval.log"

echo "Using test output dir: ${TEST_OUTPUT_DIR}"
echo "Training log:          ${TRAIN_LOG}"
echo "Evaluation log:        ${EVAL_LOG}"

# The trained agent path under the overridden output_dir
AGENT_NAME="ppo_empty_goal"
AGENT_PATH="${TEST_OUTPUT_DIR}/agents/${AGENT_NAME}"

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate /home/chen.sihe1/.conda/envs/grf_env
set -u
module load FFmpeg

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

echo "=== Starting training (empty_config) ==="
echo "Command: python3 scripts/train.py --config ${CONFIG} --override output_dir=${TEST_OUTPUT_DIR}"
echo "Training timeout: ${TRAIN_TIMEOUT}s"

# Run training with timeout, log stdout+stderr
set +e
timeout "${TRAIN_TIMEOUT}" python3 scripts/train.py \
  --config "${CONFIG}" \
  --override "output_dir=${TEST_OUTPUT_DIR}" \
  >> "${TRAIN_LOG}" 2>&1
STATUS=$?
set -e

if [[ ${STATUS} -ne 0 ]]; then
  if [[ ${STATUS} -eq 124 ]]; then
    echo "ERROR: Training timed out after ${TRAIN_TIMEOUT}s. See ${TRAIN_LOG}."
  else
    echo "ERROR: Training failed with exit code ${STATUS}. See ${TRAIN_LOG}."
  fi
  exit 1
fi

echo "Training completed successfully. See ${TRAIN_LOG}."

if [[ ! -f "${AGENT_PATH}.zip" ]]; then
  echo "ERROR: Expected trained agent not found at ${AGENT_PATH}.zip"
  exit 1
fi

# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

echo "=== Starting evaluation (empty_config) ==="
echo "Command: python3 scripts/evaluate.py --config ${CONFIG} --agent ${AGENT_PATH} --logdir ${VIDEO_DIR}"

set +e
python3 scripts/evaluate.py \
  --config "${CONFIG}" \
  --agent "${AGENT_PATH}" \
  --logdir "${VIDEO_DIR}" \
  >> "${EVAL_LOG}" 2>&1
STATUS=$?
set -e

if [[ ${STATUS} -ne 0 ]]; then
  echo "ERROR: Evaluation failed with exit code ${STATUS}. See ${EVAL_LOG}."
  exit 1
fi

echo "Evaluation completed. See ${EVAL_LOG}."

# ----------------------------------------------------------------------
# Result check: did we get the expected reward?
# ----------------------------------------------------------------------

if grep -q "Episode finished. Total reward: 1.0" "${EVAL_LOG}"; then
  echo "=== TEST PASSED ==="
  echo "Found 'Episode finished. Total reward: 1.0' in ${EVAL_LOG}"
  exit 0
else
  echo "=== TEST FAILED ==="
  echo "Did not find 'Episode finished. Total reward: 1.0' in ${EVAL_LOG}"
  exit 1
fi
