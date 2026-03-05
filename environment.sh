#!/bin/bash
# Environment setup for GRF training and evaluation

# Off-screen rendering for GRF (no display on compute nodes)
unset DISPLAY

# Setup Conda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate /home/chen.sihe1/.conda/envs/grf_env
set -u

# Load necessary modules
module load FFmpeg
