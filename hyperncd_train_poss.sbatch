#!/bin/bash
# Linea base: HyperNCD original sobre SemanticPOSS (particion 0).
#
# Uso:
#   Prueba corta (30 min, 1 epoca) para comprobar que los lotes avanzan:
#     EPOCAS=1 sbatch --time=00:30:00 hyperncd_train_poss.sbatch
#   Corrida completa (10 epocas):
#     sbatch hyperncd_train_poss.sbatch
#
# Cambios respecto de la version del 24/09, que se colgo en "Epoch 0: 0/747"
# durante 4 h (trabajo 52528):
#   - Pide una RTX A6000 completa. Con --gres=gpu:1 Slurm le dio un trozo MIG
#     de una A100 con solo 4,8 GB.
#   - Baja los procesos de lectura de 8 a 2 (WORKERS). Con 8, el DataLoader
#     puede bloquearse en el clúster sin dar error.
#   - Mas tiempo (12 h) y salida sin buffer, para ver el avance en el log.

#SBATCH --job-name=hyperncd-poss
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -e

REPO=/home/renzo.felix/PFC1-HyperNCD-Sonata
EPOCAS=${EPOCAS:-10}
WORKERS=${WORKERS:-2}

module purge
module load cuda/11.8
module load gnu12
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hyperncd-env

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export SETUPTOOLS_USE_DISTUTILS=stdlib
export PYTHONUNBUFFERED=1   # que el avance aparezca en el log al momento

echo "=== $(date)  epocas=$EPOCAS  workers=$WORKERS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO/hyperncd"

# El dataset debe descargarse antes, desde el login node (los nodos GPU no
# tienen salida a internet). Aqui solo se verifica que ya exista.
if [ ! -d data/SemanticPOSS ]; then
  echo "ERROR: data/SemanticPOSS no existe. Descargalo antes desde el login node con:"
  echo "  python scripts/download_semanticposs.py"
  exit 1
fi

python train.py -s 00 --dataset SemanticPOSS --offline \
  --epochs "$EPOCAS" --num_workers "$WORKERS" --use_scheduler \
  --lam 1 --lam_region 1 --gamma 1 --alpha 1 --gamma_decrease 0.5 \
  --smooth_bound 10 --ak_bound 0.005 --dbscan 0.5

echo "=== fin $(date)"
