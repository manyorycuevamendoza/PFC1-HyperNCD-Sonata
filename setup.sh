#!/usr/bin/env bash
# Setup para el proyecto PFC1: HyperNCD + Sonata.
# Clona ambos repos y crea DOS entornos conda separados (son incompatibles
# entre sí: CUDA/PyTorch distintos). Ver README.md para el detalle.
set -e

HYPERNCD_URL="https://github.com/2490o/HyperNCD.git"
SONATA_URL="https://github.com/facebookresearch/sonata.git"

echo "=== PFC1: setup de HyperNCD + Sonata ==="
read -p "¿Dónde estás corriendo esto? [1] Laptop  [2] Khipu (cluster UTEC): " ENV_CHOICE

# 1. Clonar ambos repos si no existen ya en esta carpeta
if [ ! -d hyperncd ]; then
  git clone "$HYPERNCD_URL" hyperncd
else
  echo "hyperncd/ ya existe, no se vuelve a clonar."
fi

if [ ! -d sonata ]; then
  git clone "$SONATA_URL" sonata
else
  echo "sonata/ ya existe, no se vuelve a clonar."
fi

if [ "$ENV_CHOICE" = "2" ]; then
  echo ""
  echo "--- Estás en Khipu: revisa/ajusta los module load ---"
  echo "Módulos de cuda/anaconda disponibles (ajusta según lo que veas):"
  module avail 2>&1 | grep -i -E "cuda|anaconda|miniconda" || echo "  (no se detectó 'module' o no hay resultados; carga conda manualmente)"
  module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || echo "  -> ejecuta manualmente: module load <tu-modulo-de-conda>"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda no está en el PATH. Instálalo (laptop) o carga el módulo correspondiente (Khipu) antes de continuar."
  exit 1
fi

# 2. Entorno para HyperNCD — sigue el setup de NOPS (python3.8 / cuda11.3 / torch1.10.1)
echo ""
echo "--- Creando entorno hyperncd-env (python 3.8) ---"
conda create -y -n hyperncd-env python=3.8
conda run -n hyperncd-env pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html
conda run -n hyperncd-env pip install pytorch-lightning==1.4.8 torchmetrics==0.7.2 scipy==1.7.3 wandb
echo ""
echo "  MinkowskiEngine necesita compilarse contra la CUDA toolkit 11.3 del sistema."
echo "  Con 'conda activate hyperncd-env' activo, sigue:"
echo "  https://github.com/NVIDIA/MinkowskiEngine#installation"

# 3. Entorno para Sonata
echo ""
echo "--- Creando entorno sonata-env ---"
if [ -f sonata/environment.yml ]; then
  conda env create -n sonata-env -f sonata/environment.yml --verbose
else
  conda create -y -n sonata-env python=3.10
  echo "  No se encontró sonata/environment.yml — instala en modo 'package':"
  echo "  revisa la sección 'package mode' en sonata/README.md (spconv, torch-scatter, flash-attention)."
fi

echo ""
echo "=== Listo ==="
echo "Repos en ./hyperncd y ./sonata."
echo "Activa cada entorno con:"
echo "  conda activate hyperncd-env"
echo "  conda activate sonata-env"
