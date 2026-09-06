# PFC1 — HyperNCD + Sonata

Proyecto de tesis: integrar el *encoder* auto-supervisado de **Sonata** en el
módulo de prototipos de **HyperNCD** (sustituir el descriptor geométrico
analítico `f_geo` por `f_ssl`), conservando íntegro el razonamiento por
hipergrafo. Corresponde al objetivo **OE2** del plan de tesis.

## Repos originales

- **HyperNCD** (CVPR 2026): https://github.com/2490o/HyperNCD
  Paper: https://arxiv.org/abs/2606.07280
- **Sonata** (CVPR 2025 Highlight): https://github.com/facebookresearch/sonata
  Paper: https://arxiv.org/abs/2503.16429
  Codebase completo de preentrenamiento (por si necesitas reproducir el
  pretraining, no solo inferencia): https://github.com/Pointcept/Pointcept

HyperNCD no trae su propio archivo de dependencias: su README remite al setup
de **NOPS** (https://github.com/LuigiRiz/NOPS), que es el paper base de la
línea (Riz et al. 2023, ya citado en tu Capítulo I).

## ⚠️ Importante: son dos entornos distintos, no uno

Las dos pilas de CUDA/PyTorch son incompatibles entre sí — no vas a poder
instalar ambos repos en el mismo `conda env`:

| | HyperNCD (vía NOPS) | Sonata |
|---|---|---|
| Python | 3.8 | según `environment.yml` (recomendado 3.10) |
| CUDA | 11.3 | 12.4 |
| PyTorch | 1.10.1 | 2.5.0 |
| Backbone | MinkowskiEngine (MinkowskiUNet-34C) | Point Transformer V3 (spconv) |
| Otras deps clave | PyTorch Lightning 1.4.8, torchmetrics 0.7.2, SciPy 1.7.3, wandb | torch-scatter, flash-attention, huggingface_hub, timm, open3d |

`setup.sh` crea **dos** entornos conda separados: `hyperncd-env` y
`sonata-env`. La integración real que pide tu OE2
(`F_i = [f_sem(x^i); f_ssl(x^i)]`) no se puede hacer importando ambos
paquetes en el mismo proceso: la forma práctica es correr Sonata congelado en
`sonata-env`, cachear sus features a disco (una pasada por sección, como ya
dice tu propuesta), y cargar ese cache desde `hyperncd-env` al construir el
hipergrafo. Esto además resuelve por diseño la diferencia de resolución
PTv3 / rejilla de Minkowski: el *up-cast* se hace una sola vez al cachear, no
en cada paso de entrenamiento.

## Uso

```bash
bash setup.sh
```

El script pregunta si estás en tu laptop o en Khipu, clona ambos repos (si no
existen ya) y crea los dos entornos conda. La instalación de MinkowskiEngine
(compilación desde código, requiere la CUDA toolkit 11.3 del sistema) y la
de Sonata en "package mode" quedan con instrucciones impresas en pantalla
porque dependen de la CUDA que tengas disponible en cada máquina.

Estructura resultante:

```
PFC1-HyperNCD-Sonata/
├── hyperncd/                    (clon de github.com/2490o/HyperNCD)
├── sonata/                      (clon de github.com/facebookresearch/sonata)
├── setup.sh
├── khipu_job_example.sbatch
└── README.md
```

## En tu laptop

Sirve para leer/editar código y probar Sonata en modo demo si tu laptop tiene
GPU NVIDIA. Entrenar HyperNCD de verdad (SemanticKITTI/SemanticPOSS o tus
secciones de la Catedral) muy probablemente necesita más VRAM de la que trae
una laptop — para eso está Khipu.

## En Khipu (cluster Slurm de UTEC)

`khipu_job_example.sbatch` es una plantilla, **no una copia exacta** de la
configuración real de Khipu (esta sesión no tiene acceso a Khipu ni a tus
credenciales). Antes de lanzarla:

1. Corre `sinfo` para ver el nombre real de la partición GPU.
2. Corre `module avail` para ver los módulos de CUDA/Anaconda disponibles y
   ajusta los `module load` del script.
3. Ajusta `--gres=gpu:*`, `--time` y `--mem` según lo que tengas asignado
   (tu propuesta ya nota que la config de GPU del clúster cambió
   recientemente, así que verifica antes de fijar tiempos).

## Citas

```bibtex
@inproceedings{riz2023nops,
  author    = {Riz, Luigi and Saltori, Cristiano and Ricci, Elisa and Poiesi, Fabio},
  title     = {Novel Class Discovery for {3D} Point Cloud Semantic Segmentation},
  booktitle = {CVPR},
  year      = {2023}
}
@article{wu2025sonata,
  author  = {Wu, Xiaoyang and others},
  title   = {Sonata: Self-Supervised Learning of Reliable Point Representations},
  journal = {arXiv preprint arXiv:2503.16429},
  year    = {2025}
}
@inproceedings{zhang2026hyperncd,
  title     = {Geometric-Aware Hypergraph Reasoning for Novel Class Discovery in Point Cloud Segmentation},
  booktitle = {CVPR},
  year      = {2026}
}
```
