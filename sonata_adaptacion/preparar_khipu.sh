#!/bin/bash
# Prepara TODO lo de la adaptacion de Sonata en el NODO DE LOGIN (tiene internet y no gasta GPU):
# entorno, Pointcept, extension compilada, dataset y pesos base.
# Se corre UNA vez:   bash sonata_adaptacion/preparar_khipu.sh
# Si algo falla, corrigelo y vuelve a correrlo: salta lo que ya esta hecho.
set -e

REPO=${REPO:-$HOME/PFC1-HyperNCD-Sonata}
POINTCEPT=${POINTCEPT:-$HOME/Pointcept}
AQUI="$REPO/sonata_adaptacion"
ARCHS=${ARCHS:-"8.0;8.6"}        # A100 (8.0) y RTX A6000 (8.6)

cd "$REPO"
if [ ! -d data ] || [ -z "$(ls data/*.npy 2>/dev/null)" ]; then
  echo "ERROR: no hay data/*.npy en $REPO (salida del Modulo 1)."; exit 1
fi

# 1. Entorno: copia de sonata-env (no se toca el original)
source "$AQUI/entorno.sh" sonata-env
if conda env list | awk '{print $1}' | grep -qx pointcept-env; then
  echo "[1/7] pointcept-env ya existe."
else
  echo "[1/7] Clonando sonata-env -> pointcept-env (unos minutos)..."
  conda create -y -n pointcept-env --clone sonata-env
fi
source "$AQUI/entorno.sh" pointcept-env

# CUDA del entorno (sonata-env trae nvcc 12.4 de conda)
export CUDA_HOME="$CONDA_PREFIX"
if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
  echo "ERROR: no hay nvcc en $CUDA_HOME/bin. Corre 'module avail 2>&1 | grep -i cuda' y haz"
  echo "       'module load cuda/12.x' antes de este script, luego exporta CUDA_HOME."; exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
TV=$(python -c "import torch;print(torch.__version__.split('+')[0])")
CU=$(python -c "import torch;print(torch.version.cuda.replace('.',''))")
echo "      python $(python -V 2>&1 | cut -d' ' -f2) | torch $TV | cu$CU | nvcc $(nvcc --version | tail -1)"

# 2. Dependencias de Pointcept
echo "[2/7] Dependencias..."
pip install -q addict yapf einops h5py plyfile termcolor tensorboardX tensorboard wandb SharedArray \
    ftfy regex tqdm scipy pyyaml packaging torch-geometric
# transformers/peft recientes exigen PyTorch >= 2.6; sonata-env trae 2.5.0, asi que se fijan versiones compatibles
pip install -q "transformers==4.46.3" "peft==0.13.2"
pip install -q torch-cluster -f "https://data.pyg.org/whl/torch-${TV}+cu${CU}.html" \
  || { echo "No hay rueda de torch-cluster; compilando (~10 min)..."; \
       TORCH_CUDA_ARCH_LIST="$ARCHS" FORCE_CUDA=1 pip install -q --no-build-isolation torch-cluster; }

# 3. Pointcept
echo "[3/7] Pointcept..."
[ -d "$POINTCEPT" ] || git clone --depth 1 https://github.com/Pointcept/Pointcept "$POINTCEPT"

# 4. pointops (extension CUDA) para A100 y A6000
echo "[4/7] Compilando pointops para sm_${ARCHS}..."
if python -c "import pointops" 2>/dev/null; then
  echo "      ya instalado."
else
  (cd "$POINTCEPT/libs/pointops" && TORCH_CUDA_ARCH_LIST="$ARCHS" FORCE_CUDA=1 \
     pip install -q --no-build-isolation --no-deps .)
fi

# 5. Parche de atencion, dataset y pesos base
echo "[5/7] Parche de atencion, dataset y pesos base..."
python "$AQUI/adaptar.py" parchear --pointcept "$POINTCEPT"
python "$AQUI/adaptar.py" datos
python "$AQUI/adaptar.py" pesos

# 6. Comprobacion de modulos (si alguno falla, el script se detiene)
echo "[6/7] Importando modulos..."
export PYTHONPATH="$POINTCEPT:$AQUI/stubs"
FALLOS=0
for m in torch spconv.pytorch torch_scatter pointops torch_cluster torch_geometric peft transformers timm addict pointcept.models pointcept.datasets; do
  if (cd "$POINTCEPT" && python -c "import $m" 2>/tmp/err_$$); then echo "   OK  $m"; else echo "   FALLA $m: $(tail -1 /tmp/err_$$)"; FALLOS=$((FALLOS+1)); fi
done
rm -f /tmp/err_$$
if [ "$FALLOS" -gt 0 ]; then
  echo "ERROR: $FALLOS modulo(s) no se pueden importar. Corrigelos y vuelve a correr este script."; exit 1
fi

# 7. Config de prueba + claves de los pesos
echo "[7/7] Verificando que los pesos de Sonata encajan en el modelo de Pointcept..."
python "$AQUI/adaptar.py" config --pointcept "$POINTCEPT" \
    --salida "$POINTCEPT/configs/sonata/pretrain-sonata-chequeo.py"
(cd "$POINTCEPT" && python "$AQUI/adaptar.py" verificar --pointcept "$POINTCEPT" \
    --config configs/sonata/pretrain-sonata-chequeo.py --pesos "$AQUI/pesos_base/sonata_init.pth") \
  || { c=$?; [ $c -eq 3 ] && { echo "ERROR: las claves no coinciden."; exit 3; } || echo "AVISO: no se pudo verificar aqui (sin GPU?); lo hace el job."; }

echo
echo "LISTO. Siguiente paso: prueba corta (gasta ~20 min de GPU):"
echo "  cd $REPO && mkdir -p logs && PRUEBA=1 sbatch --time=00:30:00 sonata_adaptacion/sonata_adaptacion.sbatch"
