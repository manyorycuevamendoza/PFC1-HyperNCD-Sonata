# Carpeta DATA: muestra del escaneo PL7

Copia local de **un solo escaneo** (PL7) del levantamiento de la iglesia, para
ver con tus propios ojos cómo es el dato antes y después del preprocesamiento
(Módulo 1 de la tesis). Los 23 escaneos completos están en Khipu, en
`~/PFC1-HyperNCD-Sonata/datos_crudos/`.

---

## Los archivos, uno por uno

### 1. `PL7_SCN0001.las` (279 MB): el escaneo original
- Es el archivo tal como lo entrega el escáner láser (GLS-2000).
- **8 590 904 puntos.** Cada punto tiene posición (x, y, z), color (R, G, B),
  intensidad del láser, clase y otros campos del formato LAS.
- Las coordenadas están en **UTM** (x ≈ 278 783, y ≈ 8 667 754), es decir, en
  metros sobre el mapa de Perú; `z` es la altura sobre el nivel del mar.
- El color está en **16 bits** (0–65535).
- La clase es **0 en todos los puntos**: el dato no trae etiquetas.
- Es **binario**: el Bloc de notas no puede leerlo. Se abre con CloudCompare.

### 2. `PL7_muestra_100.csv`: 100 puntos del original, en texto
- Tabla legible del `.las`: una fila por punto, columnas
  `x, y, z, intensity, red, green, blue, classification, return_number, gps_time`.
- Se abre con el Bloc de notas o con Excel.

### 3. `PL7_SCN0001_preprocesado.npy` (21 MB): lo que recibe Sonata
- Resultado de `preprocesar.py` sobre el `.las`:
  1. **Limpieza de ruido (SOR):** quita los puntos aislados (406 485 puntos, 4.7 %).
  2. **Voxelización a 5 cm:** deja un punto por cada cubo de 5 cm, con la
     posición y el color promediados.
  3. **Normales:** calcula hacia dónde mira la superficie en cada punto.
- **610 637 puntos** (14 veces menos que el original), 9 columnas:
  `x y z  R G B  nx ny nz`.
- Las coordenadas están **centradas en 0** (ya no en UTM) y el color en 0–255.
- Es un arreglo de NumPy (binario): se lee con Python (`numpy.load`).

### 4. `PL7_SCN0001_preprocesado.json`: la "ficha" del preprocesamiento
- Texto. Se abre con el Bloc de notas.
- Guarda **los parámetros usados** (vóxel 5 cm, q = 20 vecinos, λ = 2) y
  **cuántos puntos quedaron en cada paso** (`n_crudo`, `n_sor`, `n_voxel`).
- Guarda también `offset_utm`: lo que se le restó a las coordenadas al
  centrarlas. Sumándolo se vuelve a la posición real en UTM.

### 5. `PL7_SCN0001_preprocesado.ply` (30 MB): el preprocesado para verlo en 3D
- Los mismos 610 637 puntos del `.npy`, en un formato que abren los visores 3D
  (CloudCompare, MeshLab), con color y normales.
- Si lo abres junto al `.las` aparecerán en lugares distintos: el `.ply` está
  centrado en 0 y el `.las` en UTM. La forma es la misma.

### 6. `PL7_preprocesado_muestra_100.csv`: 100 puntos del preprocesado, en texto
- Tabla legible del `.npy`, columnas `x, y, z, red, green, blue, nx, ny, nz`.
- Se abre con el Bloc de notas o con Excel.

---

## ¿Por qué las muestras tienen solo 100 puntos?

1. **El archivo completo no se puede leer como texto.** El original tiene
   8.6 millones de puntos. En CSV ocuparía cerca de 700 MB y el Bloc de notas se
   cuelga mucho antes. Excel tampoco sirve: su límite es 1 048 576 filas.
2. **Para entender la estructura bastan unas pocas filas.** Todas las filas
   tienen las mismas columnas. Con 100 se ve qué campos hay, en qué unidades
   están y qué valores típicos toman.
3. **Los 100 puntos no son los primeros 100: están repartidos por todo el
   archivo** (uno cada ~86 000 en el original y uno cada ~6 000 en el
   preprocesado). Así la muestra recorre todo el escaneo y no solo una esquina.
4. **La muestra es solo para mirar, no para trabajar.** Con 100 puntos no se
   ve la forma de la iglesia. Para ver la nube completa se usa el `.las` o el
   `.ply` en CloudCompare, y para Sonata se usa el `.npy` completo.
