#!/bin/bash
#SBATCH --job-name=demo-sonata
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -x

REPO=/home/renzo.felix/PFC1-HyperNCD-Sonata

module purge
module load miniconda/3.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sonata-env

nvidia-smi

cd "$REPO/sonata"
export PYTHONPATH="$REPO/sonata"
python demo/2_sem_seg.py