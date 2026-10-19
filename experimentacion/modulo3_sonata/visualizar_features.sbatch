#!/bin/bash
# Dibuja y mide las features de Sonata (Modulo 3). Solo usa CPU.
# Uso (desde la raiz del repo):
#       sbatch experimentacion/modulo3_sonata/visualizar_features.sbatch PL13_SCN0001 PL13
#       (nombre del .npy en data/ y prefijo del .npz en cache/)

#SBATCH --job-name=visfeats
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -e

REPO=/home/renzo.felix/PFC1-HyperNCD-Sonata
DATO=${1:-PL13_SCN0001}
FEATS=${2:-PL13}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sonata-env

cd "$REPO"
mkdir -p experimentacion/resultados logs

python experimentacion/modulo3_sonata/visualizar_features.py \
    --feats "cache/${FEATS}_feats.npz" --data "data/${DATO}.npy" \
    --output "experimentacion/resultados/modulo3_${FEATS}" --html
