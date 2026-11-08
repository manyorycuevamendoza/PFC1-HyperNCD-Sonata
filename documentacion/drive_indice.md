# Índice del Google Drive del proyecto

Carpeta: **`PFC1_HyperNCD_Sonata`** (Mi unidad de `renzofelixaponte@gmail.com`). Una carpeta por módulo.
Los cuadernos de Colab crean y ordenan esta estructura solos (celda 1), sin pisar archivos.

```
PFC1_HyperNCD_Sonata/
├── LEEME_DRIVE.md                       este índice (copia de documentacion/drive_indice.md)
├── m1/
│   ├── entrada/                         (reservada: escaneos crudos si se suben)
│   └── salida/data_m1.tar.gz            carpeta data/ de Khipu (23 .npy + .json del Módulo 1)
├── m2/
│   ├── entrada/datos_m2_descomprimidos/hyperncd/checkpoints/SemanticPOSS/
│   │       split0_run4/SemanticPOSS_S0_best-epoch=8-step=11196.ckpt   checkpoint de HyperNCD que se usa
│   └── salida/resultados_m2/            f_sem de PL7 y PL13 y resultados de M2 (colab_runner.ipynb)
├── m3/
│   ├── entrenamiento/                   adaptación de Sonata (colab_sonata_adaptacion.ipynb)
│   │   ├── model/                       epoch_5, 10, 15, 20, 25, 30 y model_last.pth (1,8 GB c/u)
│   │   └── train.log                    registro de Pointcept (pérdida y tiempos por época)
│   ├── pesos/sonata_catedral.pth        pesos adaptados en formato sonata.load
│   └── salida/                          PL7_SCN0001_adaptado_feats.npz, PL13_SCN0001_adaptado_feats.npz
├── m4/salida/                           (reservada: grupos y métricas de la fusión)
└── entornos/                            entornos compilados para no reinstalar
    ├── pointcept_env_py<...>_torch<...>_cu<...>.tar.gz      (Sonata/Pointcept en Colab)
    └── hyperncd_env_cache.tar.gz                            (HyperNCD / MinkowskiEngine en Colab)
```

Notas:
- Los entornos compilados dependen de la GPU de la sesión: si cambias de tipo de GPU, hay que
  recompilar las extensiones (`torch-scatter`, `torch-cluster`, `pointops`).
- Si ves una carpeta `model_prueba_10it` dentro de `m3/entrenamiento/`, es la corrida inválida
  (10 iteraciones por época, por un error de `loop`). No sirve para comparar y se puede borrar.
- Fuera de esta carpeta hay material de la asesoría (papers, `Otros.txt`, propuesta) en
  "Manyory C./ Renzo F." compartido por el asesor; no lo toca el pipeline.
- Nada de Drive va a git. La copia local de lo necesario está en `experimentacion/datos/` y `cache/`.
