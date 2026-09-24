#!/bin/bash
#SBATCH --job-name=hyperncd-poss
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -x

REPO=/home/renzo.felix/PFC1-HyperNCD-Sonata

module purge
module load miniconda/3.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hyperncd-env

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

nvidia-smi

cd "$REPO/hyperncd"

# Descarga el dataset una sola vez (se salta si data/SemanticPOSS ya existe).
python scripts/download_semanticposs.py

python train.py -s 00 --dataset SemanticPOSS --offline --epoch 10 --use_scheduler \
  --lam 1 --lam_region 1 --gamma 1 --alpha 1 --gamma_decrease 0.5 \
  --smooth_bound 10 --ak_bound 0.005 --dbscan 0.5
