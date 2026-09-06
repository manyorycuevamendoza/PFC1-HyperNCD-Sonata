#!/bin/bash
# Plantilla de ejemplo para Khipu (Slurm). NO es una copia de la config real
# del clúster -- ajusta cada línea marcada con AJUSTA antes de lanzarla.
#SBATCH --job-name=hyperncd-sonata
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu            # AJUSTA: nombre real de la partición (ver `sinfo`)
#SBATCH --gres=gpu:1                # AJUSTA: cantidad/tipo de GPU asignada
#SBATCH --time=04:00:00             # AJUSTA: tiempo real que necesites
#SBATCH --mem=32G                   # AJUSTA: según RAM del nodo
#SBATCH --cpus-per-task=8

# AJUSTA los nombres de módulo según lo que muestre `module avail` en Khipu
module load anaconda3
module load cuda/11.3

mkdir -p logs

# --- Paso 1 (una sola vez por sección): cachear features de Sonata ---
# conda activate sonata-env
# python sonata/demo/extraer_features.py --input <seccion.ply> --output cache/<seccion>_feats.npy

# --- Paso 2: entrenar/evaluar HyperNCD usando el cache de Sonata ---
conda activate hyperncd-env
python hyperncd/train.py -s 00 --dataset SemanticKITTI --offline --epoch 10 \
  --use_scheduler --lam 1 --lam_region 1 --gamma 1 --alpha 1 \
  --gamma_decrease 0.5 --smooth_bound 10 --ak_bound 0.005 --dbscan 0.5
