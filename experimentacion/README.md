# Experimentación

Pruebas, revisiones visuales y figuras sobre los resultados reales del pipeline.
El código principal (`preprocesar.py`, `extraer_features.py`) vive en la raíz;
aquí solo hay herramientas para **mirar y medir** lo que esos programas producen.

> ¿Solo quieres ver y entender la vista 3D? Lee
> [VER_RESULTADOS_SONATA.md](VER_RESULTADOS_SONATA.md).

## Estructura

```
experimentacion/
├── visor_3d.py                ← VISOR PRINCIPAL. Un escaneo en todas sus etapas,
│                                misma cámara, exporta PNG. Úsalo para ver y para figuras.
├── modulo3_sonata/
│   ├── visualizar_features.py   medidas de f_ssl (rango efectivo, varianza PCA,
│   │                            k-means) en .json + imagen Matplotlib de 4 vistas
│   └── visualizar_features.sbatch   lo mismo, como trabajo en Khipu
├── figuras_informe/
│   └── componer_lamina.py     une capturas del visor en una lámina con títulos
├── datos/                     ← NO va a git (pesado). Ver "Datos" abajo.
│   ├── modulo1_PL7/
│   └── modulo3_PL13/
└── resultados/
    ├── visor_PL7.html          vista 3D de PL7: crudo y Módulo 1   (NO va a git)
    ├── visor_PL13.html         vista 3D de PL13: crudo, M1 y M3    (NO va a git)
    ├── modulo3_PL13.json       medidas de f_ssl (visualizar_features.py)
    ├── modulo3_PL13.png        4 vistas Matplotlib (visualizar_features.py)
    └── informe/                capturas del visor y láminas para la tesis
        ├── PL7_*.png, PL13_*.png             una captura por etapa, mismo ángulo
        ├── lamina_PL7_entrada_M1.png         → tesis: imagenes/visor_pl7_entrada_m1.png
        ├── lamina_PL13_entrada_M1_M3.png     → tesis: imagenes/visor_pl13_entrada_m1_m3.png
        └── lamina_PL13_descriptores_M3.png   → tesis: imagenes/visor_pl13_descriptores_m3.png
```

## Datos (`datos/`, no están en git)

| Archivo | Qué es | De dónde sale |
|---|---|---|
| `modulo1_PL7/PL7_SCN0001.las` | Escaneo crudo PL7 (8 590 904 puntos, UTM) | Khipu `datos_crudos/` |
| `modulo1_PL7/PL7_SCN0001_preprocesado.npy/.json/.ply` | Salida del Módulo 1 para PL7 | `preprocesar.py` en la laptop |
| `modulo1_PL7/*muestra_100.csv`, `README.md` | 100 puntos legibles y explicación | laptop |
| `modulo3_PL13/PL13_SCN0001.las` | Escaneo crudo PL13 (8 239 904 puntos). En Khipu se llama `PL13'_SCN0001.las` (con apóstrofo); aquí se renombró | Khipu `datos_crudos/` |
| `modulo3_PL13/PL13_SCN0001.npy` | Salida del Módulo 1 **generada en Khipu**: es la que entró a Sonata | Khipu `data/` |
| `modulo3_PL13/PL13_SCN0001.json` | Metadatos (offset UTM y conteos) | `preprocesar.py` en la laptop, mismos conteos que Khipu |
| `modulo3_PL13/PL13_feats.npz` | Salida de Sonata: 796 315 × 1088 | Khipu `cache/` |

### ⚠️ Regla importante: el `.npz` de Sonata va con SU `.npy`

Los descriptores de Sonata se asignan a los puntos **por el orden de las filas**
del `.npy`. Preprocesar el mismo escaneo con otra versión de Open3D da **los
mismos puntos en otro orden** (comprobado con PL13: Khipu usa Open3D 0.19 y la
laptop 0.20; los 803 099 puntos coinciden, el orden no). Si se mezclan, los
colores del Módulo 3 caen en puntos equivocados y la PCA parece ruido.

- Usa siempre el `.npy` **que entró a `extraer_features.py`** (el de Khipu `data/`).
- `visor_3d.py` lo comprueba y se detiene con un aviso si no corresponden.
- El `.json` sí se puede regenerar en otra máquina: offset y conteos son idénticos.

## Ver y generar figuras

Desde la raíz del repo:

```
python experimentacion/visor_3d.py \
    --las   experimentacion/datos/modulo3_PL13/PL13_SCN0001.las \
    --npy   experimentacion/datos/modulo3_PL13/PL13_SCN0001.npy \
    --json  experimentacion/datos/modulo3_PL13/PL13_SCN0001.json \
    --feats experimentacion/datos/modulo3_PL13/PL13_feats.npz \
    --output experimentacion/resultados/visor_PL13.html
```

- Sin `--feats` solo salen las etapas del Módulo 1 (así se hizo `visor_PL7.html`).
- `--camara X,Y,Z` fija el ángulo inicial (default `1.25,-1.25,0.95`, oblicua
  desde arriba; `0,0,2.2` = planta; `2.2,0,0.3` = de costado).
- Abre el `.html` con **doble clic**. Gira, elige el ángulo y pulsa la cámara 📷
  (arriba a la derecha) para bajar el PNG. Al cambiar de botón el ángulo **no**
  cambia, así que todas las etapas salen con el mismo encuadre.
- Para la lámina: `figuras_informe/componer_lamina.py --imagenes ... --titulos ... --output ...`
  (ejemplo en el propio archivo).

Las capturas de `resultados/informe/` se exportaron con la cámara por defecto y
100 000 puntos por etapa (semilla 0).

## Por qué un visor propio y no Matplotlib

Matplotlib dibuja 3D en la CPU proyectando sobre un plano: con cientos de miles
de puntos es lento, resuelve mal qué punto tapa a cuál y entrega una imagen fija
con un ángulo decidido de antemano. Por eso la figura del cubo de la tesis
(`LATEX/asesoria5/imagenes/prueba_pl7_cubo.png`) muestra solo 1 m. El visor usa
Plotly (WebGL, GPU del navegador): gira con fluidez con 100 000 puntos por
etapa, mantiene la misma cámara entre etapas y es un solo archivo que cualquiera
abre sin Python.

## Historial de cambios que pueden confundir

- **`prueba_pl7_cubo.png` aparece modificada en git** (commit "cambio de la
  imagen"): se regeneró al completar el capítulo 3 para comprobar sus cifras.
  Es la misma figura: mismo cubo (x −19 a −18, y −3 a −2, z −12 a −11), mismos
  48 098 → 921 puntos; solo cambian ~2,5 KB de compresión. El cubo no es al azar:
  `figura_entrada_salida.py` elige uno con muchas superficies en varias
  orientaciones, para que se note el efecto.
- **Antes había `modulo1_preprocesamiento/visualizar_preprocesamiento.py` y
  `resultados/modulo1_*.html`, `resultados/modulo3_PL13.html`:** se reemplazaron
  por `visor_3d.py` y `visor_*.html`, que juntan todas las etapas en una vista.
- **La figura de Sonata de la tesis cambió:** `imagenes/sonata_pl13_features.png`
  (Matplotlib, 4 vistas) se reemplazó por las láminas del visor
  (`visor_pl13_entrada_m1_m3.png`, `visor_pl13_descriptores_m3.png`).
- **Se intentó exportar PNG con kaleido** (exportador automático de Plotly) y se
  cuelga en Windows; por eso las capturas salen del botón 📷 del propio visor.
