#!/bin/bash
# Paso 1 del pipeline: cachear features SSL de Sonata.
# Uso:  sbatch extraer_features.sbatch data/Otros.txt cache/Otros_feats.npz

#SBATCH --job-name=feats
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -e

REPO=/home/renzo.felix/PFC1-HyperNCD-Sonata
ENTRADA=${1:-data/Otros.txt}
SALIDA=${2:-cache/Otros_feats.npz}

module purge
module load miniconda/3.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sonata-env

cd "$REPO"
mkdir -p cache

# Los nodos de computo no tienen salida a internet: los pesos ya estan
# cacheados en ~/.cache (descargados desde el nodo de login).
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO/sonata"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python extraer_features.py --input "$ENTRADA" --output "$SALIDA"
