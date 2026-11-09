# Cómo correr todo el pipeline (M1 → M2 → M3 → M4) en Colab con una escena etiquetada

Un solo cuaderno: [`colab_pipeline_etiquetado.ipynb`](../colab_pipeline_etiquetado.ipynb). Corre los cuatro
módulos sobre una escena con etiquetas y mide mIoU. Los tiempos de esta guía son **estimados, no medidos en
Colab**: el cuaderno se probó solo en sus partes que corren en la laptop (unir, M1, mapeo de etiquetas y
evaluación). M2 y M3 no se han corrido aún con él.

## 1. Qué hay que tener antes

| Qué | Dónde (Drive: `PFC1_HyperNCD_Sonata/`) |
|---|---|
| Escena etiquetada: `listo.zip` (un `.las` por pieza; la clase es el nombre del archivo) | `m1/entrada/etiquetado/listo.zip` (o la celda 4 te deja subirlo) |
| Checkpoint de HyperNCD (M2) | `m2/entrada/...` (ya está el de la época 8; otro modelo: súbelo y pon su ruta en `CKPT_M2`) |
| Pesos de Sonata adaptado (M3) | `m3/pesos/sonata_catedral.pth` (ya está) |
| El repo con estos archivos **subidos a GitHub** (rama `main`) | `colab_pipeline_etiquetado.ipynb` y `experimentacion/evaluacion_etiquetada/` |

El cuaderno clona tu repo, así que lo nuevo tiene que estar en GitHub antes de correr la celda 2 (`git add`,
`git commit`, `git push`).

## 2. Pasos

1. Abre `colab_pipeline_etiquetado.ipynb` en Colab (desde GitHub o súbelo a Drive) y elige **GPU**.
2. Celda 1: revisa los **parámetros** (`ESCENA`, `CKPT_M2`, `M3_VARIANTES`, `K_LISTA`).
3. Corre las celdas **en orden, de la 1 a la 10**. La 11 (control) es opcional.
4. Lee la tabla de la celda 10 y mira `evaluacion_miou.png`.
5. Todo queda en Drive, en `m4/salida/<ESCENA>/`.

**Correr solo un M3** (por ejemplo solo el adaptado): deja una sola entrada en `M3_VARIANTES`.
**Otro modelo:** agrega una línea a `M3_VARIANTES` con la ruta de sus pesos, o cambia `CKPT_M2`. Las demás
partes (M1, M2) no se repiten si ya están en Drive.

## 3. ¿Cómo funcionan los entornos en Colab?

**Colab no guarda nada entre sesiones.** Cada vez que conectas, la máquina virtual está vacía: `/content` se
borra y la GPU puede ser otra. Lo que sí persiste es **tu Drive**. Por eso los entornos se guardan allí como
archivos comprimidos:

| Entorno | Para qué | Archivo en Drive (`entornos/`) | Cómo se restaura |
|---|---|---|---|
| `pcenv` (Sonata + Pointcept) | M1, M3 | `pointcept_env_<python>_<torch>_<cuda>.tar.gz` | celda 3 |
| `hyperncd-env` (Python 3.8 + MinkowskiEngine) | M2 | `hyperncd_env_cache.tar.gz` | celda 6 |

**Cómo "se da cuenta" el sistema:** no hay magia. Cada celda calcula el nombre esperado del archivo y mira
si existe en Drive:
- Si **existe**, lo descomprime en `/content` (1 a 2 minutos; ~20 s el de M2) y listo.
- Si **no existe**, construye el entorno (15 a 25 minutos) y lo guarda para la próxima vez.

El nombre del entorno de Sonata incluye las versiones de Python, PyTorch y CUDA de la sesión. Si Google
actualiza la imagen de Colab y esas versiones cambian, el nombre cambia y se reconstruye **una vez**.

**Qué pasa si cambias de GPU (T4, L4, A100...):** las extensiones compiladas (`torch-scatter`,
`torch-cluster`, `pointops`) dependen de la arquitectura de la GPU. La celda **3b** lo prueba y, si fallan, las
recompila para esa GPU (5 a 10 min) y **actualiza** el archivo de Drive. Esa fue la causa del error
`no kernel image` que tuvimos. El entorno de M2 se compiló para las arquitecturas 7.5, 8.0 y 8.6; debería
servir en la L4 por compatibilidad entre versiones menores, **pero no lo he comprobado**, y no servirá en
GPU más nuevas (H100).

**Resumen:** la primera vez en cada tipo de GPU es lenta. Después, restaurar es rápido, y es lo que esperas:
"correr solo".

## 4. Qué guarda en Drive

```
m1/entrada/<ESCENA>/        escena unida (.las) y etiquetas (.npz)
m1/salida/<ESCENA>/         M1: .npy, .json y etiquetas por punto (_etiquetas.npy/.json)
m2/salida/<ESCENA>/         f_sem (.npz, .json, .csv)
m3/salida/<ESCENA>/         f_ssl de cada variante (<ESCENA>_feats_<variante>.npz)
m4/salida/<ESCENA>/         m4_<variante>/ y evaluacion/ (evaluacion.md, .json, evaluacion_miou.png)
```

Si se cae la sesión, vuelve a correr desde la celda 1: lo ya calculado se copia de Drive y no se repite
(`REHACER = True` fuerza recalcular, por ejemplo si cambias el checkpoint de M2 o los pesos de Sonata).

## 5. Lo que se sabe del dataset `listo`

- 19 `.las`, **una sola escena** de 3 380 661 puntos únicos, unos 14 × 23 × 20 m, con cúpula.
- **`otro.las` trae toda la escena.** Las demás piezas son subconjuntos repetidos. El cuaderno da prioridad
  a la clase específica sobre `otro`.
- Reparto de puntos: otro 61,1 %, pared 15,6 %, cúpula 9,1 %, techo 6,7 %, arco 4,3 %, columna 3,2 %.
- 23 811 puntos (0,7 %) aparecen en dos clases específicas distintas (por ejemplo en un borde): se conserva una.
- **No está dentro de PL7 ni de PL13.** El punto más cercano de esos escaneos queda a unos 10 m. Es otra
  parte de la capilla: sirve como **escena nueva con etiquetas**, no como etiquetas de PL7 o PL13.
- Tras el M1 quedan ~1,14 millones de puntos, y cada uno recibe la etiqueta del punto crudo más cercano
  (distancia mediana 1,2 cm).

## 6. Cómo leer la evaluación

- Los grupos de M4 no tienen nombre. Se reporta mIoU asignando cada grupo a su clase mayoritaria y también
  con asignación 1 a 1 (húngara).
- **Referencia:** asignar todo a "otro" da mIoU 0,10 y exactitud 0,61. Un resultado que no la supere no dice nada.
- **Tope por el tamaño de región:** con regiones de 0,75 m, incluso un agrupamiento ideal por región (simulado)
  da mIoU 0,86. Las regiones mezclan piezas en los bordes.
- **Cobertura:** M4 solo agrupa los puntos que M2 marca como novedosos. Lo que M2 marca como clase conocida no
  se evalúa; la tabla dice qué fracción quedó fuera.
- Una sola corrida, sin barras de error. Con diferencias pequeñas entre variantes no se puede concluir.

## 7. Si algo falla

| Síntoma | Qué hacer |
|---|---|
| `Falta ... evaluacion_etiquetada/...` en la celda 2 | Esos archivos no están en `main`: commit y push |
| Celda 3: `ENTORNO INCOMPLETO` | Copia el módulo que falla; si es una copia mala en `entornos/`, bórrala y vuelve a correr |
| Celda 3b: sigue fallando tras recompilar | Pega el error completo |
| `OutOfMemory` en M3 | Las celdas ya aplican la atención SDPA; si persiste, usa una GPU con más memoria |
| `MemoryError` en M4 | Usa RAM alta (Colab Pro) o sube `--celda` a 1,0 en la celda 9 |
| M2 da `AssertionError` de coordenadas | El `.npy` y el `f_sem` no corresponden: `REHACER = True` |

## 8. Pendientes

- Probar el cuaderno completo en Colab (M2 y M3 no se han corrido aún con él).
- Registrar los resultados en `experimentacion/registro/` cuando haya cifras.
- Control de la adaptación (celda 11): mide el cambio del descriptor solo por los pesos, en el mismo entorno.
