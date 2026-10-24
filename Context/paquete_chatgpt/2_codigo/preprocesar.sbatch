#!/bin/bash
# Modulo 1 del pipeline: preprocesar los .las crudos (SOR + voxel + normales).
# Solo usa CPU. Procesa cada .las de datos_crudos/ como una seccion y deja
# data/<nombre>.npy listo para extraer_features.sbatch.
#
# Uso:  sbatch preprocesar.sbatch                # todos los .las
#       sbatch preprocesar.sbatch "PL4*"         # solo los que calcen
#       VOXEL=0.03 LAM=2.5 sbatch preprocesar.sbatch   # cambiar parametros

#SBATCH --job-name=preproc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -e

REPO=/home/renzo.felix/PFC1-HyperNCD-Sonata
PATRON=${1:-*}
VOXEL=${VOXEL:-0.05}
Q=${Q:-20}
LAM=${LAM:-2.0}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sonata-env

cd "$REPO"
mkdir -p data logs

find datos_crudos -name "${PATRON}.las" | sort | while read -r las; do
    nombre=$(basename "$las" .las | tr ' ' '_' | tr -d "'")
    if [ -f "data/${nombre}.npy" ] && [ -z "$REHACER" ]; then
        echo "[skip] data/${nombre}.npy ya existe"
        continue
    fi
    echo "=== $las -> data/${nombre}.npy"
    python preprocesar.py --input "$las" --output "data/${nombre}" \
        --voxel "$VOXEL" --q "$Q" --lam "$LAM"
done
