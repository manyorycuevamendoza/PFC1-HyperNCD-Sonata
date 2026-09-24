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
module load cuda/11.8
module load gnu12
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hyperncd-env

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export SETUPTOOLS_USE_DISTUTILS=stdlib

nvidia-smi

cd "$REPO/hyperncd"

# El dataset debe descargarse antes, desde el login node (los nodos GPU no
# tienen salida a internet). Aquí solo se verifica que ya exista.
if [ ! -d data/SemanticPOSS ]; then
  echo "ERROR: data/SemanticPOSS no existe. Descárgalo antes desde el login node con:"
  echo "  python scripts/download_semanticposs.py"
  exit 1
fi

python train.py -s 00 --dataset SemanticPOSS --offline --epoch 10 --use_scheduler \
  --lam 1 --lam_region 1 --gamma 1 --alpha 1 --gamma_decrease 0.5 \
  --smooth_bound 10 --ak_bound 0.005 --dbscan 0.5
