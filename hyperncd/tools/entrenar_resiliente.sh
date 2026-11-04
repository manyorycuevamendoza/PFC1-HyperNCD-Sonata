#!/bin/bash
# Lanza train.py y, si se cae (p. ej. "launch timed out" de CUDA), espera y reanuda
# desde last.ckpt de su carpeta de checkpoints. Hasta MAX_INTENTOS veces.
#
# Uso (desde hyperncd/):
#   tools/entrenar_resiliente.sh <comment> [--resume <ckpt>] [más argumentos de train.py]
#
# Ejemplo:
#   tools/entrenar_resiliente.sh split0_run5 -s 0 --dataset SemanticPOSS --offline --num_workers 0 --batch_size 2

set -u

PY=/home/renzo/miniconda3/envs/hn2/bin/python
MAX_INTENTOS=10
ESPERA_SEGUNDOS=60

COMMENT=$1
shift
CKPT_DIR=checkpoints/SemanticPOSS/$COMMENT
ARGS=("$@")

for intento in $(seq 1 "$MAX_INTENTOS"); do
    extra=()
    if [ -f "$CKPT_DIR/last.ckpt" ]; then
        extra=(--resume "$PWD/$CKPT_DIR/last.ckpt")
    fi
    echo "[$(date '+%F %T')] intento $intento: ${extra[*]:-desde cero}"
    "$PY" -u train.py "${ARGS[@]}" --comment "$COMMENT" "${extra[@]}"
    if [ $? -eq 0 ]; then
        echo "[$(date '+%F %T')] entrenamiento terminado"
        exit 0
    fi
    echo "[$(date '+%F %T')] falló; esperando ${ESPERA_SEGUNDOS}s antes de reanudar"
    sleep "$ESPERA_SEGUNDOS"
done

echo "[$(date '+%F %T')] se agotaron los $MAX_INTENTOS intentos"
exit 1
