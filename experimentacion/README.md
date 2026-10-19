# Experimentación

Pruebas y análisis sobre los resultados reales del pipeline, separados del
código principal (`preprocesar.py`, `extraer_features.py`, que viven en la raíz).

```
experimentacion/
├── datos/                      ← NO va a git (pesado). Copia local para analizar.
│   ├── modulo1_PL7/            Escaneo PL7 antes y después del Módulo 1
│   │   ├── PL7_SCN0001.las                 original (8,6 M puntos, UTM)
│   │   ├── PL7_SCN0001_preprocesado.npy    salida del Módulo 1 (610 637 × 9)
│   │   ├── PL7_SCN0001_preprocesado.json   parámetros y conteos
│   │   ├── PL7_SCN0001_preprocesado.ply    lo mismo, para CloudCompare
│   │   ├── PL7_*muestra_100.csv            100 puntos legibles (antes / después)
│   │   └── README.md                       explicación de cada archivo
│   └── modulo3_PL13/           Escaneo PL13 a la entrada y salida del Módulo 3
│       ├── PL13_SCN0001.npy                entrada de Sonata (salida del Módulo 1)
│       └── PL13_feats.npz                  salida de Sonata: f_ssl, 1088 dim. por punto
├── modulo3_sonata/             Código del análisis de las features de Sonata
│   ├── visualizar_features.py
│   └── visualizar_features.sbatch          lo mismo, como trabajo en Khipu
└── resultados/
    ├── modulo3_PL13.png        4 vistas: color, PCA, similitud, k-means
    ├── modulo3_PL13.json       medidas (rango efectivo, varianza PCA, grupos)
    └── modulo3_PL13.html       vista 3D interactiva (NO va a git, se regenera)
```

## Cómo se obtuvieron los datos

- `modulo1_PL7/`: el `.las` se bajó de Khipu y se preprocesó en la laptop con
  `preprocesar.py` (mismos parámetros que en Khipu: vóxel 5 cm, q = 20, λ = 2).
- `modulo3_PL13/`: los dos archivos se bajaron de Khipu
  (`data/PL13_SCN0001.npy` y `cache/PL13_feats.npz`, trabajo `feats_55001`).

## Ver los resultados

**Imagen fija:** abrir `resultados/modulo3_PL13.png`.

**Vista 3D interactiva (girar, zoom, cambiar coloración):** abrir
`resultados/modulo3_PL13.html` con doble clic. Se abre en el navegador; necesita
internet la primera vez (carga Plotly). Botones de arriba: color real, PCA,
similitud con el punto ◆ y k-means.

## Regenerar

Desde la raíz del repo, en la laptop:

```
python experimentacion/modulo3_sonata/visualizar_features.py \
    --feats experimentacion/datos/modulo3_PL13/PL13_feats.npz \
    --data  experimentacion/datos/modulo3_PL13/PL13_SCN0001.npy \
    --output experimentacion/resultados/modulo3_PL13 --html
```

O en Khipu, leyendo directo de `cache/` y `data/`:

```
sbatch experimentacion/modulo3_sonata/visualizar_features.sbatch PL13_SCN0001 PL13
```

Opciones útiles: `--consulta N` fija el punto de consulta de la similitud,
`--k N` cambia el número de grupos de k-means.
