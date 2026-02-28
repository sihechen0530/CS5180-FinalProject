#!/bin/bash
# run_test.sh
#
# Quick sanity check for the empty_config pipeline:
# - Runs training and evaluation into a new test_outputs/<timestamp> directory
# - Fails if training times out (>15 min) or eval reward != 1.0
# - Prints the test output directory path when finished (for merge readiness checks)

set -euo pipefail

# ----------------------------------------------------------------------
# Configurable parameters
# ----------------------------------------------------------------------

CONFIG="configs/empty_config.yaml"
TRAIN_TIMEOUT=900

# Unique test output directory: test_outputs/<timestamp>
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
TEST_OUTPUT_DIR="test_outputs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${TEST_OUTPUT_DIR}"

LOG_DIR="${TEST_OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TRAIN_LOG="${LOG_DIR}/train.log"
EVAL_LOG="${LOG_DIR}/eval.log"

echo "Test output directory: ${TEST_OUTPUT_DIR}"
echo "Training log:          ${TRAIN_LOG}"
echo "Evaluation log:       ${EVAL_LOG}"

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate /home/chen.sihe1/.conda/envs/grf_env
set -u
module load FFmpeg
unset DISPLAY

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

echo "=== Starting training (empty_config) ==="
echo "Command: python3 scripts/train.py --config ${CONFIG} --output-dir ${TEST_OUTPUT_DIR}"
echo "Training timeout: ${TRAIN_TIMEOUT}s"

# Run training with timeout, log stdout+stderr
set +e
timeout "${TRAIN_TIMEOUT}" python3 scripts/train.py \
  --config "${CONFIG}" \
  --output-dir "${TEST_OUTPUT_DIR}" \
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

if [[ ! -f "${TEST_OUTPUT_DIR}/agents/ppo_empty_goal.zip" ]]; then
  echo "ERROR: Expected trained agent not found at ${TEST_OUTPUT_DIR}/agents/ppo_empty_goal.zip"
  exit 1
fi

# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

echo "=== Starting evaluation (empty_config) ==="
echo "Command: python3 scripts/evaluate.py --config ${CONFIG} --output-dir ${TEST_OUTPUT_DIR}"

set +e
python3 scripts/evaluate.py \
  --config "${CONFIG}" \
  --output-dir "${TEST_OUTPUT_DIR}" \
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
  echo ""
  echo "Test output directory: ${REPO_ROOT}/${TEST_OUTPUT_DIR}"
  echo "(logs, checkpoints, tensorboard, videos, and agents are under this directory)"
  exit 0
else
  echo "=== TEST FAILED ==="
  echo "Did not find 'Episode finished. Total reward: 1.0' in ${EVAL_LOG}"
  echo ""
  echo "Test output directory: ${REPO_ROOT}/${TEST_OUTPUT_DIR}"
  exit 1
fi
