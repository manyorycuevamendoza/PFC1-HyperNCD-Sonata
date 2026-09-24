# Setup en Khipu (cluster Slurm de UTEC)

Pasos ya probados y funcionando para dejar `hyperncd-env` listo para entrenar.
Sigue el orden — cada paso corrige un problema real que nos salió al hacerlo
la primera vez.

## 0. Clonar el repo

```bash
git clone git@github.com:manyorycuevamendoza/PFC1-HyperNCD-Sonata.git
cd PFC1-HyperNCD-Sonata
```

## 1. Crear el entorno conda de HyperNCD

Corre esto en el **nodo de login** (el prompt normal `[usuario@khipu ...]$`,
sin `salloc` ni `sbatch` — necesitamos internet, que los nodos GPU no tienen).

```bash
conda create -y -n hyperncd-env python=3.8
conda activate hyperncd-env
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning==1.4.8 torchmetrics==0.7.2 scipy==1.7.3 wandb pyyaml
```

## 2. Descargar el dataset (SemanticPOSS)

```bash
cd hyperncd
python scripts/download_semanticposs.py
cd ..
```

Deja los datos en `hyperncd/data/SemanticPOSS/dataset/sequences/00../05/`, que
es justo lo que pide `hyperncd/config/semposs_dataset.yaml`. No requiere
login ni registro (a diferencia de SemanticKITTI, que sí lo pide — si lo
necesitas más adelante, avisa).

## 3. Instalar MinkowskiEngine

Este es el paso que más dio problemas. Notas importantes:

- **No uses `module load miniconda/3.0`** — carga un conda del sistema
  distinto al tuyo (`/opt/ohpc/pub/apps/miniconda-3`), y no encuentra
  `hyperncd-env`. Usa tu `conda activate` normal.
- **Compílalo en el nodo de login**, no en un job de Slurm — los nodos GPU
  de Khipu no tienen salida a internet, así que `pip`/`git clone` fallan ahí.
  Compilar la extensión CUDA no necesita una GPU físicamente presente, solo
  el compilador (`nvcc`).
- **CUDA del `module load` debe coincidir en versión mayor con la de
  PyTorch** (`cu113` = CUDA 11.x). Usa `cuda/11.8` (no hay `11.3` exacto en
  Khipu, y `11.4` puede no estar disponible según el nodo — revisa con
  `module avail 2>&1 | grep -i cuda`). No uses CUDA 12.x, falla por
  incompatibilidad de versión mayor.
- **pip moderno ya no soporta `--install-option`** (lo quitaron en pip
  23.1+). Por eso instalamos con `setup.py install` directo, clonando el
  repo aparte (no dentro de este repo — es una dependencia de terceros, no
  código nuestro).

```bash
cd ~
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine

conda activate hyperncd-env
module load cuda/11.8
export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6"

conda install -y openblas-devel -c anaconda

python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

python -c "import MinkowskiEngine as ME; print('MinkowskiEngine OK, version:', ME.__version__)"
```

`TORCH_CUDA_ARCH_LIST` compila para todas las arquitecturas de GPU de Khipu
(A100, RTX A6000, y algunas más comunes) para que el mismo build sirva sin
importar qué GPU te toque en el job de entrenamiento.

Si ves `MinkowskiEngine OK, version: ...` al final, ya puedes borrar
`~/MinkowskiEngine` sin problema (ya quedó instalado dentro del entorno
conda).

## 4. Ver qué GPU hay disponible (opcional, solo para elegir partición)

```bash
sinfo -p gpu -o "%P %N %G %t"
```

En este cluster hay A100 (particionado en MIG), RTX A6000 y Tesla. Los
`.sbatch` de este repo piden `--gres=gpu:1` (genérico) para que Slurm
asigne cualquiera libre y no haya que esperar por un tipo específico.

## 5. Lanzar el entrenamiento

```bash
sbatch hyperncd_train_poss.sbatch
squeue -u $USER
```

El log queda en `logs/hyperncd-poss_<JOBID>.out`.

## Entorno de Sonata (aparte, no relacionado a estos pasos)

`sonata-env` se crea independiente siguiendo `sonata/environment.yml` (ver
`README.md` principal). Es un entorno de CUDA/PyTorch totalmente distinto e
incompatible con `hyperncd-env` — no mezclar.
