#!/bin/bash
# Activa un entorno conda en Khipu. Uso:  source entorno.sh <nombre-del-entorno>
# Prueba primero tu conda propio (~/miniconda3) y, si ahi no esta el entorno,
# el del modulo del sistema (miniconda/3.0), que es donde vive sonata-env.
_ENV_NAME="${1:-pointcept-env}"

_tiene_env() { conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$1"; }

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
if ! command -v conda >/dev/null 2>&1 || ! _tiene_env "$_ENV_NAME"; then
  module load miniconda/3.0 2>/dev/null || true
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
if ! _tiene_env "$_ENV_NAME"; then
  echo "ERROR: no encuentro el entorno conda '$_ENV_NAME' (ni en ~/miniconda3 ni en el modulo miniconda/3.0)." >&2
  return 1 2>/dev/null || exit 1
fi
conda activate "$_ENV_NAME"
