#!/bin/bash
#SBATCH --job-name=install-mke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -x

REPO=/home/renzo.felix/PFC1-HyperNCD-Sonata

module purge
module load miniconda/3.0
module load cuda/11.4
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hyperncd-env

nvidia-smi
nvcc --version

conda install -y openblas-devel -c anaconda

pip install -U git+https://github.com/NVIDIA/MinkowskiEngine.git \
  --no-deps \
  --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" \
  --install-option="--blas=openblas"

python -c "import MinkowskiEngine as ME; print('MinkowskiEngine OK, version:', ME.__version__)"
