# PFC1 — HyperNCD + Sonata (Catedral de Lima)

Tesis: integrar el *encoder* auto-supervisado **Sonata** (`f_ssl`) en el módulo de prototipos de
**HyperNCD** (`f_sem`), conservando el razonamiento por hipergrafo, y medirlo sobre nubes de
puntos de la Catedral de Lima (objetivo OE2).

**¿Qué está hecho y qué falta?** → [`ESTADO_DEL_PROYECTO.md`](ESTADO_DEL_PROYECTO.md)
**¿Cómo se prepara Khipu?** → [`SETUP_KHIPU.md`](SETUP_KHIPU.md)

## Los cinco módulos y dónde vive cada uno

| Módulo | Qué hace | Dónde corre | Código |
|---|---|---|---|
| **M1** Preprocesamiento | SOR, voxel 5 cm, normales → `.npy` de 9 columnas | Khipu (CPU) | `preprocesar.py`, `preprocesar.sbatch` |
| **M2** HyperNCD | `f_sem` (96 dim.) | Colab (MinkowskiEngine no corre en la laptop); entrenar en Khipu | `predecir_hyperncd.py`, `colab_runner.ipynb`, `hyperncd_*.sbatch`, `hyperncd/` |
| **M3** Sonata | `f_ssl` (1088 dim.), con encoder congelado o adaptado | Khipu (GPU) o Colab | `extraer_features.py` / `.sbatch`, `sonata_adaptacion/`, `colab_sonata_adaptacion.ipynb`, `sonata/` |
| **M4** Fusión e hipergrafo | `F = [f_sem ; f_ssl]`, regiones, Sinkhorn | Laptop | `experimentacion/modulo4_real/` (real), `modulo4_simulado/` |
| **M5** Evaluación | mIoU sobre la Catedral | — | pendiente |

## Mapa de carpetas

```
├── README.md · ESTADO_DEL_PROYECTO.md · SETUP_KHIPU.md     empieza por aquí
├── preprocesar.* · extraer_features.* · predecir_hyperncd.py · hyperncd_*.sbatch · demo_sonata.sbatch
├── sonata_adaptacion/             adaptar Sonata a la Catedral en Khipu (preparar + sbatch)
├── colab_runner.ipynb             M2 en Colab (solo predice)   colab_sonata_adaptacion.ipynb   M3 en Colab
├── colab_experimento_m2_dominio.ipynb   ¿el colapso de M2 es del dominio?   colab_entrenamiento_arch.ipynb   entrenar M2 con ArCH, medido
├── experimentacion/               M4, visores 3D, etiquetado, resultados (cada carpeta con su README)
│   ├── registro/                  un .json por experimento (qué es, de qué viene, parámetros, costo, métricas)
│   ├── visor/index.html           VISOR: pestañas M1–M4, lista de experimentos y modo comparar (doble clic)
│   ├── adaptacion_sonata/         experimento: Sonata original vs adaptado (comparación M3, M4 y costo)
│   ├── m2_dominio/                experimento: M2 en SemanticPOSS vs capilla (colapso por dominio)
│   ├── arch_resumen/              puntos y clases por escena de ArCH
│   ├── arch_entrenamiento/        convertir ArCH a bloques y entrenar HyperNCD con él
│   └── monitor_recursos/          mide tiempo, RAM y GPU de cualquier comando
├── documentacion/
│   ├── drive_indice.md            qué hay en cada carpeta del Drive
│   ├── prompts/                   prompts listos para pegar (p. ej. pruebas de parámetros en Khipu)
│   ├── contexto/                  defensa de M1 y M3 y muestras de datos
│   ├── tesis_latex/               texto de la tesis
│   ├── papers/                    artículos de referencia
│   └── historico/                 documentos viejos (no se mantienen)
├── hyperncd/ · sonata/            código de terceros (no tocar)
└── data/ · cache/ · logs/ · datos_crudos/    datos de trabajo (NO van a git)
```

## Qué modelo y qué pesos usar

Cuál checkpoint es el correcto (M2 época 8, Sonata original y Sonata adaptado), dónde está cada uno y cómo comprobarlo: [`documentacion/modelos_y_pesos.md`](documentacion/modelos_y_pesos.md).

**Resumen rápido de pesos (todos en Drive, `PFC1_HyperNCD_Sonata/`; ninguno va a git):**

| Pesos | Ruta |
|---|---|
| M2 HyperNCD, `best-epoch=8-step=11196.ckpt` (SemanticPOSS) | `m2/entrada/datos_m2_descomprimidos/hyperncd/checkpoints/SemanticPOSS/split0_run4/` |
| M3 Sonata adaptado, `sonata_catedral.pth` | `m3/pesos/` |
| M3 Sonata original | se descarga de Hugging Face (`facebook/sonata`); no hay archivo propio |

Ojo: `m1/salida/` solo tiene salidas del módulo 1 (`data_m1.tar.gz` y `capilla_etiquetada/`), no pesos.

## Ver los experimentos

Abre `experimentacion/visor/index.html` (doble clic, sin servidor). Tiene una pestaña por etapa
(M1 a M4), la lista de experimentos de cada una con su procedencia (qué M2 y qué M3 usó un M4),
y un modo **comparar** que pone dos experimentos lado a lado y resalta lo que cambia. Los visores
3D pesados se cargan solo al pulsar su botón.

**Regla del proyecto:** todo experimento nuevo se registra en `experimentacion/registro/` (ver su
`LEEME.md`) con su documento y con la ruta de sus datos pesados; luego
`python experimentacion/registro/registrar.py` actualiza el índice. Lo que no esté registrado no
aparece en el visor ni se considera parte del trabajo.

## Dónde están los datos

Nada de esto está en git. Viven en Khipu, en `~/PFC1-HyperNCD-Sonata/`:

| Carpeta | Qué hay | Cómo se genera |
|---|---|---|
| `datos_crudos/` | los 23 `.las` originales de la Catedral | se suben a mano |
| **`data/`** | **23 escaneos ya preprocesados**: `<escaneo>.npy` (9 columnas: X Y Z R G B Nx Ny Nz) y `<escaneo>.json` (parámetros). ~626 MB en total | `sbatch preprocesar.sbatch` (M1) |
| `cache/` | salidas de M3: `<escaneo>_feats.npz` (`f_ssl`) | `sbatch extraer_features.sbatch` |
| `logs/` | salidas de Slurm (`<job>_<id>.out/.err`) | automático |

Escaneos de prueba: `PL7_SCN0001` y `PL13_SCN0001`. Los demás son `PL1`…`PL18`, `Piso_*`,
`Campanario_Stonex`, `Capilla_Veracruz`, `capilla_secreta` e `iglesia_*_stonex`.

En Google Drive (`PFC1_HyperNCD_Sonata/`) hay una carpeta por módulo: `m1/salida/data_m1.tar.gz`
es una copia de `data/`, y `m2/`, `m3/`, `m4/` guardan entradas y salidas de cada módulo.
Para crearla en Khipu: `tar -czf data_m1.tar.gz data` y bajarla con `scp`
(comandos en `experimentacion/VER_RESULTADOS_SONATA.md`).

### ArCH (dataset público de patrimonio, para entrenar M2)

ArCH Dataset, Politecnico di Torino, licencia **CC BY-NC-SA 4.0** (hay que citarlo en la tesis). 17 escenas
etiquetadas (15 `Training` + 2 `Test`), 136 millones de puntos, 9 clases útiles más `other` (que se ignora).
Se descargó por FTP temporal directo a Drive desde Colab (no pasa por la laptop):

| Qué | Ruta en Drive |
|---|---|
| `.txt` originales (~10 GB) | `arch/datos/ArCH_labelled/{Training,Test}/` |
| resumen de puntos y clases por escena | `arch/resumen/resumen_arch.md` |
| bloques de 10 m para entrenar | `arch/tiles/` (los crea `colab_entrenamiento_arch.ipynb`) |
| checkpoints, logs y mediciones del entrenamiento | `arch/entrenamiento/` |

Estado: datos descargados y resumidos; HyperNCD (M2) **entrenado 30 épocas en Colab T4 (75 min)**: mIoU de validación 0,489 y, en la capilla, la cabeza dominante baja de 98,4 % a 44,6 % (NMI 0,02 a 0,19). Detalle en `experimentacion/arch_entrenamiento/README.md`. Con 10 semillas (una escena, K = 8, mIoU 1 a 1 sin `otro`): M4 52,5 ± 2,5, M4 base 51,4 ± 1,7, M2 48,4 ± 1,1, M3 45,4 ± 8,5; la diferencia M4 contra M4 base (+1,1) cae dentro del ruido. Ver `experimentacion/modulo4_fusion/README.md`.

## Primera vez en Khipu

```bash
git clone git@github.com:manyorycuevamendoza/PFC1-HyperNCD-Sonata.git
cd PFC1-HyperNCD-Sonata && mkdir -p logs
ls data/*.npy | wc -l        # debe dar 23; si no, corre antes: sbatch preprocesar.sbatch
```

Si ya lo tienes clonado: `git pull` y sigue. El entorno de Sonata (`sonata-env`) y el de
HyperNCD (`hyperncd-env`) ya existen; si no, `SETUP_KHIPU.md` explica cómo crearlos.

## Comandos de ejecución (desde la raíz del repo en Khipu)

```bash
sbatch preprocesar.sbatch                                              # M1: .las -> data/*.npy
sbatch extraer_features.sbatch data/PL13_SCN0001.npy cache/PL13_SCN0001_feats.npz   # M3: f_ssl
sbatch hyperncd_train_poss.sbatch                                      # M2: línea base en SemanticPOSS
bash sonata_adaptacion/preparar_khipu.sh                               # M3 adaptado: preparar (1 vez, login node)
PRUEBA=1 sbatch --time=00:30:00 sonata_adaptacion/sonata_adaptacion.sbatch   # prueba de 1 época
sbatch sonata_adaptacion/sonata_adaptacion.sbatch                      # entrenamiento completo
squeue -u $USER                                                        # ver tus trabajos
tail -n 40 logs/<job>_<id>.out                                         # ver el avance
```

- **Pipeline completo con etiquetas (M1→M4 + mIoU):** `colab_1_m1_m2.ipynb` y `colab_2_m3_m4.ipynb`; guía en `documentacion/ejecutar_pipeline_colab.md`.
- **M2 sobre la Catedral:** abre `colab_runner.ipynb` en Colab (los checkpoints están en Drive, `m2/entrada/`).
- **M3 adaptado en Colab:** `colab_sonata_adaptacion.ipynb`, celdas 1 → 8.
- **M4 (laptop):** `python experimentacion/modulo4_real/correr_modulo4_real.py` (ver su README).

## Dos entornos distintos (no se pueden mezclar)

| | HyperNCD | Sonata |
|---|---|---|
| Python / CUDA / PyTorch | 3.8 / 11.3 / 1.10.1 | 3.10 / 12.4 / 2.5.0 |
| Backbone | MinkowskiEngine | Point Transformer V3 (spconv) |

Por eso Sonata corre aparte, guarda `f_ssl` en disco (`cache/`) y M4 lo lee de ahí.
`pointcept-env` (adaptación) es una copia de `sonata-env` con lo necesario para entrenar.

## Referencias

- HyperNCD (CVPR 2026): https://github.com/2490o/HyperNCD · https://arxiv.org/abs/2606.07280
- Sonata (CVPR 2025): https://github.com/facebookresearch/sonata · https://arxiv.org/abs/2503.16429
- Pointcept (entrenamiento de Sonata): https://github.com/Pointcept/Pointcept
- NOPS (base de HyperNCD): Riz et al., *Novel Class Discovery for 3D Point Cloud Semantic Segmentation*, CVPR 2023
