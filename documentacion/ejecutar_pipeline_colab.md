# Cómo correr el pipeline en Colab (dos cuadernos)

Esta es **la única guía**. Dos cuadernos, en este orden, y el segundo lee lo que dejó el primero en tu Drive:

```
 colab_1_m1_m2.ipynb                         colab_2_m3_m4.ipynb
 ┌────────────────────────────┐              ┌──────────────────────────────────────┐
 │  M1  preprocesar           │   Drive      │  M3  Sonata (f_ssl) → M4 → evaluación │
 │  M2  HyperNCD (f_sem)      │ ───────────► │  lee M1 y M2 de Drive                 │
 └────────────────────────────┘              └──────────────────────────────────────┘
   GPU donde M2 ya funcionó                     cualquier GPU de 16 GB o más (la L4)
```

Por qué dos: **M2 y M3 usan entornos distintos** y M2 (MinkowskiEngine) no corre en todas las GPU. Así cada
cuaderno se corre en la GPU que le sirve, sin confundirse. Los tiempos de abajo son **estimados**; los cuadernos
completos no se han corrido todavía en Colab (solo se probó en la laptop lo que no necesita GPU).

> Qué checkpoint y qué pesos son los correctos, con rutas y cómo comprobarlos: [`modelos_y_pesos.md`](modelos_y_pesos.md).

## 0. Antes de empezar

1. **Sube el código a GitHub** (`main`): los dos cuadernos y `experimentacion/evaluacion_etiquetada/`. Los cuadernos clonan tu repo.
2. **Sube `listo.zip`** a Drive: `PFC1_HyperNCD_Sonata/m1/entrada/etiquetado/listo.zip` (el cuaderno 1 te deja subirlo si falta).
3. Ya deben estar en Drive el checkpoint de HyperNCD (`m2/entrada/`) y, para el Sonata adaptado, `m3/pesos/sonata_catedral.pth`.

## 1. Cuaderno 1: `colab_1_m1_m2.ipynb` (M1 + M2)

- **GPU:** la con la que ya te funcionó M2 antes. Si no la recuerdas, prueba **T4**. Si M2 da
  `CUDA error ... concurrent_unordered_map`, cambia a **A100** y vuelve a correr desde la celda 1.
- Corre las celdas **de la 1 a la 8, en orden**.
- Deja en Drive: `m1/salida/<ESCENA>/` (el `.npy`, su `.json` y las etiquetas) y `m2/salida/<ESCENA>/` (`f_sem`).
- La celda 8 confirma que todo quedó guardado y te dice que sigas con el cuaderno 2.

## 2. Cuaderno 2: `colab_2_m3_m4.ipynb` (M3 + M4 + evaluación)

- **GPU:** cualquiera con 16 GB o más; **la L4 sirve**. No tiene que ser la del cuaderno 1.
- Corre las celdas **de la 1 a la 10, en orden**.
- Celda 3: trae M1 de Drive (si falta, se detiene y te dice que corras el cuaderno 1). M2 es opcional ahí.
- **Puedes ir en paralelo:** M3 solo necesita M1. Si el cuaderno 1 sigue corriendo M2, abre el 2 en **otra sesión** y corre hasta la celda 7. La celda 8 (M4) espera a que M2 esté en Drive y, si aún no está, te lo dice y la vuelves a correr. Dos sesiones a la vez gastan unidades de cómputo las dos.
- **M4 necesita M3:** la celda 7 calcula `f_ssl` (una vez por variante: original, adaptado...) y la 8 corre M4 con cada una, junto con M1 y M2.
- Celda 9: evalúa con las etiquetas. Deja todo en `m4/salida/<ESCENA>/`.
- **Cambiar de modelo:** edita `M3_VARIANTES` (otros pesos de Sonata) en la celda 1 del cuaderno 2, o `CKPT_M2` (otro checkpoint de HyperNCD) en el cuaderno 1. Si cambias pesos o checkpoint, el cuaderno lo detecta y recalcula esa parte.

Si se cae una sesión, vuelve a correr desde su celda 1: lo ya guardado se copia de Drive y no se repite (`REHACER = True` lo fuerza).

## 3. Cómo funcionan los entornos

Colab **no guarda nada entre sesiones**: `/content` se borra y la GPU puede ser otra. Lo que persiste es tu Drive, así que
los entornos se guardan allí como archivos comprimidos en `entornos/`.

| Entorno | Lo usa | Archivo en Drive | Cómo se restaura |
|---|---|---|---|
| HyperNCD (Python 3.8 + MinkowskiEngine) | cuaderno 1, M2 | `hyperncd_env_cache.tar.gz` | celda 6 (~20 s) |
| Sonata (Pointcept) | cuaderno 2, M3 y M4 | `pointcept_env_<python>_<torch>_<cuda>_sm<GPU>.tar.gz` (uno **por tipo de GPU**) | celda 4 (1-2 min) |

No hay magia: cada celda calcula el nombre esperado del archivo y mira si existe en Drive. Si existe, lo descomprime. Si no,
construye el entorno (15 a 25 min) y lo guarda. La celda 5 del cuaderno 2 comprueba que las extensiones compiladas de Sonata
funcionen **en esa GPU** y, si no, las recompila (5 a 10 min) y actualiza el archivo. La primera vez en cada tipo de GPU es lenta;
después, rápido. El cuaderno 1 no necesita el entorno de Sonata: M1 solo usa `laspy` y `open3d`, que instala la celda 3.

## 4. Qué queda en Drive (`PFC1_HyperNCD_Sonata/`)

```
m1/entrada/<ESCENA>/       escena unida (.las) y etiquetas (.npz)          ← cuaderno 1
m1/salida/<ESCENA>/        M1: .npy, .json, _etiquetas.npy/.json           ← cuaderno 1
m2/salida/<ESCENA>/        M2: f_sem (.npz, .json, .csv)                   ← cuaderno 1
m3/salida/<ESCENA>/        M3: un f_ssl por variante                       ← cuaderno 2
m4/salida/<ESCENA>/        m4_<variante>/, evaluacion/, evaluacion_sin_otro/, logs/   ← cuaderno 2
```

`logs/` guarda lo que imprime cada módulo (tiempos y VRAM), útil para la tabla de costo computacional.

## 4b. Bajar lo necesario a la laptop (cuadros, comparativas y visor)

1. **En Colab** (cuaderno 2, celda 10): arma `paquete_<ESCENA>.zip` y lo **descarga solo** a tu carpeta Descargas. Incluye M1 (nube y
   etiquetas), M2 (`f_sem`), M4 de cada variante de M3, las evaluaciones y los logs. **No incluye** los `f_ssl` (~2,4 GB cada uno).
2. **En la laptop**, desde la raíz del repo:

```bash
python experimentacion/evaluacion_etiquetada/traer_resultados.py
```

   Toma el `paquete_*.zip` más reciente de Descargas y lo deja en `experimentacion/datos/escena_<ESCENA>/` (no va a git), con la misma
   estructura que el Drive. Si tienes **Google Drive para escritorio**, sirve sin zip:
   `python experimentacion/evaluacion_etiquetada/traer_resultados.py --drive-desktop "G:/Mi unidad/PFC1_HyperNCD_Sonata"`.

## 5. Cómo leer la evaluación

- Los grupos de M4 no tienen nombre: se reporta mIoU asignando cada grupo a su clase mayoritaria y también 1 a 1 (húngara).
- **Referencia:** asignar todo a "otro" da mIoU ≈ 0,10 y exactitud ≈ 0,61. Un resultado que no la supere no dice nada.
- **Dos evaluaciones:** `evaluacion/` cuenta todas las clases; `evaluacion_sin_otro/` ignora `otro`, que es la nube completa (lo no
  segmentado) y puede contener elementos reales sin marcar. Revísalo en CloudCompare.
- **Tope por el tamaño de región:** con regiones de 0,75 m, un agrupamiento ideal por región (simulado) da mIoU ≈ 0,86.
- **Cobertura:** M4 solo agrupa los puntos que M2 marca como novedosos; lo demás no se evalúa. La tabla dice cuánto quedó fuera.
- Una corrida, sin barras de error: diferencias pequeñas entre variantes no permiten concluir.

## 6. Si algo falla

| Síntoma | Qué hacer |
|---|---|
| `Falta ... evaluacion_etiquetada/...` en la celda 2 | Ese archivo no está en `main`: commit y push |
| Cuaderno 2, celda 3: "Faltan estos archivos en Drive" | Corre antes el cuaderno 1 hasta su celda 8 |
| Cuaderno 1, M2: `CUDA error ... concurrent_unordered_map` | Cambia a una **T4 o A100** y repite desde la celda 1 |
| Cuaderno 1, celda 3: no importa `open3d` | Pega el error: puede que Colab haya cambiado de versión de Python |
| Cuaderno 2, celda 4: `ENTORNO INCOMPLETO` | Copia el módulo que falla; si es una copia mala en `entornos/`, bórrala y repite |
| Cuaderno 2, celda 5: sigue fallando tras recompilar | Pega el error completo |
| `OutOfMemory` en M3 | Usa una GPU con más memoria |
| `MemoryError` en M4 | Usa RAM alta (Colab Pro) o sube `--celda` a 1,0 en la celda 8 |

## 7. Sobre el dataset `listo`

- 19 `.las`, **una sola escena** de 3 380 661 puntos únicos (unos 14 × 23 × 20 m, con cúpula). Reparto: otro 61,1 %, pared 15,6 %,
  cúpula 9,1 %, techo 6,7 %, arco 4,3 %, columna 3,2 %.
- **`otro.las` trae toda la escena**; las demás piezas son subconjuntos repetidos. La unión da prioridad a la clase específica sobre `otro`.
- 23 811 puntos (0,7 %) aparecen en dos clases específicas distintas (bordes): se conserva una.
- **No está dentro de PL7 ni de PL13** (el punto más cercano de esos escaneos queda a unos 10 m): es una escena nueva con etiquetas reales.
- Tras M1 quedan ~1,14 millones de puntos; cada uno recibe la etiqueta del punto crudo más cercano (distancia mediana 1,2 cm).
- Para verla en CloudCompare, coloreada por clase: `experimentacion/evaluacion_etiquetada/visual/` (se regenera, no va a git).
