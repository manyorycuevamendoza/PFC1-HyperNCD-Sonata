#!/bin/bash
# Adaptacion de Sonata a la Catedral en Khipu: entrena -> exporta pesos -> extrae f_ssl de PL7 y PL13.
# Todo en UN solo job para no hacer cola dos veces. Cada etapa se salta si ya esta hecha,
# asi que reenviar el job despues de un corte NO repite trabajo ni gasta GPU de mas.
#
# Antes (una vez, en el login node):  bash sonata_adaptacion/preparar_khipu.sh
#
# Uso:
#   Prueba corta (1 epoca; comprueba memoria, velocidad y todo el recorrido):
#     PRUEBA=1 sbatch --time=00:30:00 sonata_adaptacion/sonata_adaptacion.sbatch
#   Corrida completa:
#     sbatch sonata_adaptacion/sonata_adaptacion.sbatch
#   Variantes (cada una con su propio RUN para no pisar la otra):
#     RUN=lote2 BATCH=2 MAX_PUNTOS=32768 sbatch sonata_adaptacion/sonata_adaptacion.sbatch
#   Si se corto por tiempo: el mismo comando otra vez (reanuda desde model_last.pth)
#
# Parametros (variables de entorno): RUN EPOCAS LOOP BATCH MAX_PUNTOS LR WORKERS SAVE_FREQ ESCANEOS PRUEBA

#SBATCH --job-name=sonata-adapt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -eo pipefail

REPO=${REPO:-$HOME/PFC1-HyperNCD-Sonata}
POINTCEPT=${POINTCEPT:-$HOME/Pointcept}
AQUI="$REPO/sonata_adaptacion"

PRUEBA=${PRUEBA:-0}
RUN=${RUN:-adapt_v1}
EPOCAS=${EPOCAS:-30}
LOOP=${LOOP:-5}
BATCH=${BATCH:-4}
MAX_PUNTOS=${MAX_PUNTOS:-65536}
LR=${LR:-0.0002}
WORKERS=${WORKERS:-6}
SAVE_FREQ=${SAVE_FREQ:-10}
ESCANEOS=${ESCANEOS:-"PL7_SCN0001 PL13_SCN0001"}
if [ "$PRUEBA" = "1" ]; then
  RUN=prueba; EPOCAS=1; LOOP=1; SAVE_FREQ=1; ESCANEOS="PL13_SCN0001"
fi

RUNDIR="$AQUI/runs/$RUN"
CFG_REL="configs/sonata/pretrain-sonata-$RUN.py"
PESOS_OUT="$AQUI/runs/$RUN/sonata_catedral.pth"
mkdir -p "$RUNDIR"

source "$AQUI/entorno.sh" pointcept-env
export CUDA_HOME="$CONDA_PREFIX"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1                       # los nodos de computo no tienen internet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$POINTCEPT:$AQUI/stubs"

echo "=== $(date)  nodo=$(hostname)  RUN=$RUN"
echo "    epocas=$EPOCAS loop=$LOOP batch=$BATCH max_puntos=$MAX_PUNTOS lr=$LR workers=$WORKERS save_freq=$SAVE_FREQ"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---------- Comprobaciones: si algo falta, se corta AQUI en segundos (no despues de horas) ----------
python - <<'PY'
import importlib, sys, torch
if not torch.cuda.is_available():
    sys.exit("ERROR: torch no ve la GPU")
x = torch.randn(8, device="cuda"); assert x.sum().isfinite()
malos = []
for m in ["spconv.pytorch", "torch_scatter", "pointops", "torch_cluster", "torch_geometric",
          "peft", "transformers", "timm", "addict", "pointcept.models", "pointcept.datasets"]:
    try:
        importlib.import_module(m)
    except Exception as e:
        malos.append(f"{m}: {e}")
if malos:
    sys.exit("ERROR modulos:\n  " + "\n  ".join(malos))
import torch_scatter
torch_scatter.segment_coo(torch.randn(6, device="cuda"), torch.tensor([0,0,1,1,2,2], device="cuda"), reduce="min")
print("entorno OK |", torch.cuda.get_device_name(0), "| torch", torch.__version__)
PY
[ -f "$AQUI/datos_pointcept/.completo" ] || { echo "ERROR: falta el dataset. Corre preparar_khipu.sh en el login node."; exit 1; }
[ -f "$AQUI/pesos_base/sonata_init.pth" ] || { echo "ERROR: faltan los pesos base. Corre preparar_khipu.sh en el login node."; exit 1; }
for e in $ESCANEOS; do [ -f "$REPO/data/$e.npy" ] || { echo "ERROR: no existe $REPO/data/$e.npy"; exit 1; }; done
NECESARIO=$(( (EPOCAS / SAVE_FREQ + 2) * 2 ))
LIBRE=$(df -BG --output=avail "$RUNDIR" | tail -1 | tr -dc 0-9)
echo "espacio libre ${LIBRE} GB; checkpoints necesitan ~${NECESARIO} GB (cada uno pesa ~2 GB)"
[ "$LIBRE" -ge "$NECESARIO" ] || { echo "ERROR: poco espacio en disco. Borra runs viejos o sube SAVE_FREQ."; exit 1; }

python "$AQUI/adaptar.py" config --pointcept "$POINTCEPT" --salida "$POINTCEPT/$CFG_REL" \
    --epocas "$EPOCAS" --loop "$LOOP" --batch "$BATCH" --max-puntos "$MAX_PUNTOS" \
    --lr "$LR" --workers "$WORKERS" --save-freq "$SAVE_FREQ" --amp bfloat16

# ---------- 1. Entrenar ----------
if [ -f "$RUNDIR/.entrenado" ]; then
  echo "[1/3] Entrenamiento ya terminado en este RUN: se salta."
else
  if [ -f "$RUNDIR/model/model_last.pth" ]; then
    echo "[1/3] Reanudando desde model_last.pth"
    OPC="save_path=$RUNDIR weight=$RUNDIR/model/model_last.pth resume=True"
  else
    echo "[1/3] Entrenamiento nuevo (parte de Sonata preentrenado)"
    OPC="save_path=$RUNDIR weight=$AQUI/pesos_base/sonata_init.pth resume=False"
  fi
  (cd "$POINTCEPT" && python -u tools/train.py --config-file "$CFG_REL" --num-gpus 1 --options $OPC) 2>&1 | tee -a "$RUNDIR/train.log"
  touch "$RUNDIR/.entrenado"
fi

# ---------- 2. Exportar los pesos al formato de sonata.load ----------
echo "[2/3] Exportando pesos adaptados"
python "$AQUI/adaptar.py" exportar --run "$RUNDIR" --salida "$PESOS_OUT"

# ---------- 3. Extraer f_ssl adaptado ----------
echo "[3/3] Extrayendo f_ssl adaptado"
mkdir -p "$REPO/cache"
cd "$REPO"
export PYTHONPATH="$REPO/sonata:$POINTCEPT:$AQUI/stubs"
for e in $ESCANEOS; do
  SAL="cache/${e}_adaptado_${RUN}_feats.npz"
  if [ -f "$SAL" ]; then echo "  $SAL ya existe, se salta"; continue; fi
  echo "  --- $e"
  python extraer_features.py --input "data/$e.npy" --output "$SAL" --pesos "$PESOS_OUT"
done

echo "=== fin $(date)"
echo "Pesos:    $PESOS_OUT"
echo "Features: $REPO/cache/*_adaptado_${RUN}_feats.npz"
