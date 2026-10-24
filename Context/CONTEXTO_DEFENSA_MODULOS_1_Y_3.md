# Contexto para la defensa: Módulo 1 (preprocesamiento) y Módulo 3 (Sonata)

Documento de estudio. Todo lo que dice sale del código del repositorio, de los
archivos de metadatos (`.json`) y de los resultados guardados. Cuando un dato
**solo aparece en el texto de la tesis y no tiene registro en el repo**, se marca
con **[sin log en el repo]**, para que sepas qué puedes demostrar y qué no.

---

## 1. El proyecto en una frase

HyperNCD descubre clases nuevas en nubes de puntos usando, por cada punto, un
descriptor semántico (`f_sem`, 96 valores) y un descriptor geométrico manual
(`f_geo`, 6 valores). La tesis reemplaza `f_geo` por el descriptor de Sonata
(`f_ssl`, 1088 valores), un codificador preentrenado sin etiquetas que se usa
**congelado**. Hasta hoy están ejecutados el **Módulo 1** (preprocesamiento) y el
**Módulo 3** (Sonata). Los Módulos 2, 4 y 5 (HyperNCD, fusión y evaluación) todavía no.

---

## 2. Los datos

### 2.1 El levantamiento completo
| Dato | Valor | Fuente |
|---|---|---|
| Archivos LAS | 23 | tesis [sin log en el repo] |
| Tamaño total | 20,05 GB | tesis [sin log en el repo] |
| Puntos crudos totales | 589 708 290 | tesis [sin log en el repo] |
| Representantes tras el Módulo 1 (los 23) | 17 807 859 | tesis [sin log en el repo] |
| Suma de tiempos del Módulo 1 (los 23) | 1 079 s (~18 min) | tesis [sin log en el repo] |

- Cada archivo LAS es **una estación del escáner** (PL7_SCN0001, PL13_SCN0001…), no
  un pedazo separado de la catedral. **Las estaciones se superponen**: el centro de
  PL7 y el de PL13 están a ~17 m y cada una abarca ~±26 m. Por eso los 589 millones
  de puntos incluyen superficies vistas desde varias estaciones.
- Los LAS no traen etiquetas: el campo de clase vale 0 en todos los puntos.
- Para verificar los totales en Khipu:
  ```bash
  cd /home/renzo.felix/PFC1-HyperNCD-Sonata
  ls datos_crudos/**/*.las | wc -l                 # cuántos LAS
  du -sh --apparent-size datos_crudos/            # cuánto pesan
  ls data/*.npy | wc -l                           # cuántos pasaron el Módulo 1
  cat data/*.json | grep n_voxel                  # representantes por escaneo
  ls cache/*.npz                                  # cuántos pasaron por Sonata
  ```

### 2.2 Escaneo de prueba del Módulo 1: PL7 (en la laptop)
| Etapa | Puntos | Fuente |
|---|---|---|
| Crudo (`n_crudo`) | **8 590 904** | `PL7_SCN0001_preprocesado.json` |
| Tras SOR (`n_sor`) | **8 184 419** (se descartan 406 485, el 4.73 %) | json |
| Tras voxelizar (`n_voxel`) | **610 637** (7.46 % de los que quedaron del SOR) | json |
| Tiempo total | 14.4 s | json |
| Archivo LAS | 292 090 963 B (292 MB) | disco |
| Matriz S (`.npy`) | 21 983 060 B (22 MB), 610 637 × 9 en float32 | disco |
| Centro restado (UTM) | 278800.97, 8667752.42, 174.10 | json |

Equipo: laptop AMD Ryzen 5 8645HS (6 núcleos, 12 hilos), 16 GB de RAM, Windows 11,
Python 3.12.7, laspy 2.7.0, NumPy 2.3.5, Open3D 0.20.0. La GPU (RTX 3050) no se usa.

### 2.3 Escaneo de prueba del Módulo 3: PL13 (en Khipu)
**Sonata se ejecutó sobre un solo escaneo de los 23: PL13.** Es el 1,4 % de los
puntos del levantamiento.

| Etapa | Puntos | Fuente |
|---|---|---|
| Crudo | **8 239 904** | `PL13_SCN0001.json` |
| Tras SOR | **8 020 796** | json |
| Tras voxelizar = entrada a Sonata | **803 099** × 9 | json |
| Tras la rejilla de 2 cm de Sonata | **796 315** (99.2 %) | `modulo3_PL13.json` |
| Salida `f_ssl` | 796 315 × 1088 | `modulo3_PL13.json` |
| Archivo de descriptores `.npz` | 680 139 814 B (680 MB) | disco |
| Tiempo de inferencia | 13,9 s | tesis [sin log en el repo] |
| Rendimiento | 57 397 puntos/s | tesis [sin log en el repo] |
| VRAM pico | 34,60 GB de 48 GB (72 %) | tesis [sin log en el repo] |
| Parámetros del modelo | 108,46 M | log `logs/demo-sonata_*.out` y recálculo capa por capa = 108 461 328 |

- En Khipu el archivo se llama `PL13'_SCN0001.las` (con apóstrofo). El guion le
  quita el apóstrofo al nombrar la salida: `data/PL13_SCN0001.npy`.
- La matriz S de PL13 que entró a Sonata se generó **en Khipu** (Open3D 0.19).
- Medidas sin etiquetas de `f_ssl` (`modulo3_PL13.json`):
  - 3 primeras componentes principales: 25.3 % de la varianza; 10 primeras: 43.4 %.
  - Rango efectivo: 110.6 sobre 1088 (1 = colapso, 1088 = reparto igual).
  - k-means con k = 8, tamaños de los grupos: 96 835, 161 418, 112 804, 46 961, 79 436, 98 247, 142 162, 58 452.

**Conviene guardar en `logs/` el `.out` de la corrida de Sonata en Khipu**: ahí están
el tiempo, la VRAM y el rendimiento que hoy no tienen respaldo en el repo.

---

## 3. ¿Hubo paralelismo? (respuesta honesta)

### 3.1 En el Módulo 1 (preprocesamiento de los 23 archivos)
- **No hubo paralelismo entre archivos.** `preprocesar.sbatch` lanza **un solo
  trabajo** de Slurm y dentro recorre los LAS **de uno en uno** con un `while`.
- Recursos del trabajo: 16 núcleos de CPU, 64 GB de RAM, sin GPU, límite de 4 h.
- El único paralelismo es **interno de Open3D** (el filtro SOR, la voxelización y
  las normales usan varios hilos en los 16 núcleos). No se midió cuánto aporta.
- Si un archivo ya tiene su `.npy`, el guion lo salta: un trabajo cortado se retoma.
- **Se podría** lanzar un trabajo por archivo (*job array*), porque las secciones
  son independientes, pero no hizo falta: los 23 tardaron ~18 min en total.

### 3.2 En el Módulo 3 (Sonata)
- **Un trabajo por escaneo**, en una GPU RTX A6000 de 48 GB, 8 núcleos, 64 GB de RAM,
  límite de 1 h. Solo se lanzó para PL13.
- **No hubo paralelismo entre escaneos ni entre GPU.**
- **Sí hay paralelismo dentro de la GPU**: la nube serializada se corta en parches
  de 1024 puntos y la atención de **todos los parches y todas las cabezas** se
  calcula a la vez, como una operación por lotes.
- La profundidad (5 etapas en orden) **no se puede repartir** entre GPU: cada etapa
  necesita la salida de la anterior.
- Con varias GPU, la aceleración máxima es `A = min(n_gpu, n_secciones)`: una
  sección no se reparte entre GPU. Con 23 secciones, más de 23 GPU no ayudan.

### 3.3 Por qué el disco está en medio
- Sonata (PTv3) necesita CUDA 12.4; MinkowskiEngine (HyperNCD) necesita CUDA 11.x.
  **No pueden convivir en el mismo proceso.** Por eso hay dos entornos conda
  (`sonata-env` y `hyperncd-env`) y `f_ssl` se escribe en un `.npz`.
- Además Sonata está congelado: su salida no cambia entre épocas. Se calcula
  **una vez por sección** y HyperNCD la lee en todas las épocas.

---

## 4. Procedimiento del Módulo 1, paso a paso (`preprocesar.py`)

Comando de la prueba de PL7:
```bash
python preprocesar.py --input PL7_SCN0001.las --output PL7_SCN0001_preprocesado \
       --voxel 0.05 --q 20 --lam 2
```

1. **Carga por bloques** (`leer_las`). laspy lee el LAS en bloques de 5 millones
   de puntos. De cada punto toma `x, y, z` y `R, G, B`. Descarta intensidad,
   retorno y clase. El color viene en 16 bits (máx. 65 535) y se divide entre 256
   para llevarlo a 0–255. Todos los bloques se unen en memoria: **la sección
   completa debe caber en la RAM**.
2. **Centrado.** Las coordenadas UTM son números del orden de 10^6 m; en float32
   perderían precisión. Se resta el centro de la caja,
   `offset = (min + max) / 2` por eje, y se guarda en el `.json` para poder volver
   a UTM. La forma no cambia.
3. **Filtrado SOR** (`remove_statistical_outlier`, q = 20, λ = 2). Para cada punto,
   distancia media `d_p` a sus 20 vecinos. Con la media μ y desviación σ de todos
   los `d_p`, se conserva el punto si `d_p ≤ μ + 2σ`. Va **antes** de voxelizar
   para que el ruido no se promedie con puntos buenos. En PL7 descarta el 4,73 %.
   Ojo: elimina también parte de superficies lejanas y ralas (umbral único).
4. **Voxelización** (`voxel_down_sample`, ℓ = 5 cm). Divide el espacio en cubos de
   5 cm; los puntos de cada cubo se sustituyen por **uno**, con la posición y el
   color **promedio**. Open3D cuenta los cubos desde la esquina mínima de la nube
   menos ℓ/2. Iguala la densidad (cerca del escáner hay muchos más puntos). No es
   limpieza: no borra superficies, reduce cuántos puntos las describen. Límite:
   superficies a menos de 5 cm se funden (se ve en el cubo de 1 m de PL7:
   48 098 → 921).
5. **Normales** (`estimate_normals`, KNN = 20). Covarianza de los 20 vecinos de cada
   representante; la normal es el autovector del **menor** autovalor. Se orientan
   con `n_z ≥ 0` (`orient_normals_to_align_with_direction([0,0,1])`).
6. **Escritura.** `S = [x y z R G B nx ny nz]` en float32 → `.npy`. Metadatos
   (parámetros, offset, `n_crudo`, `n_sor`, `n_voxel`, segundos) → `.json`.

**Reproducibilidad.** No hay pasos aleatorios: misma entrada → mismos puntos.
**Pero el orden de las filas puede cambiar** con otra versión de Open3D (comprobado
con PL13: Khipu 0.19 y laptop 0.20 dan los mismos 803 099 puntos en distinto orden).
Como Sonata asocia descriptores por orden de filas, **cada `.npz` solo sirve con el
`.npy` exacto que entró a Sonata**.

---

## 5. Procedimiento del Módulo 3, paso a paso (`extraer_features.py`)

Comando en Khipu:
```bash
sbatch extraer_features.sbatch data/PL13_SCN0001.npy cache/PL13_feats.npz
```

1. **Carga de la matriz S** (`cargar_nube`). Lee el `.npy` (803 099 × 9) y lo separa
   en `coord`, `color` y `normal` (cada una N × 3), sin cambiar el orden.
2. **Carga del modelo** (`cargar_modelo`). `sonata.load("sonata")` lee los pesos
   públicos (descargados antes desde el nodo de acceso: los nodos GPU no tienen
   internet, por eso `HF_HUB_OFFLINE=1`). Configuración: **FlashAttention
   desactivada** (no está instalada en Khipu) y **parches de 1024 puntos**. Se pasa a
   GPU en modo evaluación: los pesos no cambian. 108,46 M de parámetros.
3. **Preparación** (`sonata.transform.default()`), en este orden:
   - **CenterShift**: resta el centro de la caja en x, y, y el mínimo en z.
   - **GridSample** (2 cm, modo `train`): divide en celdas de 2 cm y se queda con
     **un punto al azar por celda**. Guarda `grid_coord` (celda entera de cada punto)
     e **`inverse`** (a qué celda cayó cada punto de S). En PL13: 803 099 → 796 315
     (99,2 %), porque la celda de 2 cm es menor que el vóxel de 5 cm del Módulo 1.
     Por el azar, dos corridas no dan exactamente el mismo `f_ssl`.
   - **NormalizeColor**: color / 255.
   - **ToTensor y Collect**: arma `feat` = [coord, color, normal] → M × 9.
4. **Inferencia** (`torch.inference_mode()`, sin gradientes):
   - **Embedding**: lineal 9 → 48 + normalización + GELU.
   - **Serialización**: ordena los puntos con curvas que llenan el espacio (Z y
     Hilbert) para que vecinos en 3D queden vecinos en la secuencia.
   - **5 etapas** con 3, 3, 3, 12 y 3 bloques y 48, 96, 192, 384 y 512 canales.
     Cada **bloque**: convolución posicional (CPE, convolución dispersa 3×3×3),
     atención multi-cabeza dentro de parches de 1024 (Q, K, V), y MLP (×4).
   - **Pooling entre etapas**: la celda se **duplica** (2 → 4 → 8 → 16 → 32 cm); una
     capa lineal sube los canales, y por celda se toma el **máximo** de las
     características y el **promedio** de las coordenadas. Guarda
     `pooling_parent` y `pooling_inverse` para poder volver.
   - **Up-cast** (función `upcast`): desde la etapa 5 hacia arriba. En los dos
     primeros pasos **concatena**: etapa 5 (512) con la 4 (384) → 896, y eso con la
     3 (192) → **1088**. Luego **copia** a las etapas 2 y 1 por índice. Resultado:
     un vector de 1088 por cada uno de los 796 315 puntos de la rejilla.
5. **Escritura**: `.npz` comprimido con `feat` (float16, la mitad de tamaño),
   `coord`, `inverse`, `grid_size` y `n_original`. En PL13: 680 MB.

**Vuelta a los puntos de S**: `f_ssl_por_punto = feat[inverse]` → 803 099 × 1088.
Así cada punto de S recibe el descriptor de su celda. Es lo que usa el visor y lo
que usará HyperNCD en la fusión.

**Revisión de los descriptores** (`experimentacion/modulo3_sonata/visualizar_features.py`
y `experimentacion/visor_3d.py`, semilla 0, muestra de 100 000 puntos):
- PCA a 3 componentes usada como color.
- Similitud coseno con un punto de consulta.
- k-means (k = 8, 30 iteraciones) sobre descriptores centrados y normalizados.
- Rango efectivo = exp(−Σ p_k ln p_k) con las varianzas normalizadas.
- **Nada de esto es una evaluación de acierto**: no hay etiquetas. La evaluación
  prevista es el *linear probing* (`ŷ = argmax(Wᵀ f_ssl + b)`), que requiere anotar.

### 5.1 Por qué 34,60 GB de VRAM si la cota decía 4,2 GB
- La cota se calculó para 3 × 10^5 puntos y suponía que cada etapa reduce los puntos
  a la mitad. La prueba tenía 796 315 puntos.
- Sin FlashAttention la atención se guarda completa en memoria por parche.
- **La causa exacta no se ha verificado.** Si la memoria crece con los puntos, los
  escaneos más grandes podrían no caber: se prevé partir la sección en bloques con
  solape.

---

## 6. Preguntas probables del profesor (y respuestas)

**¿Corrieron Sonata sobre todo el levantamiento?**
No. Solo sobre PL13 (803 099 puntos de entrada, 796 315 tras la rejilla). Es una
prueba de funcionamiento y de costo. El Módulo 1 sí se ejecutó sobre los 23 LAS.

**¿Usaron paralelismo?**
Entre archivos, no: el Módulo 1 los procesó en serie en un solo trabajo de 16
núcleos (~18 min los 23), y Sonata se lanzó una vez, en una GPU. Dentro de cada
proceso sí: Open3D usa varios hilos, y en la GPU la atención de todos los parches y
cabezas se calcula a la vez. Como las secciones son independientes, se pueden lanzar
en paralelo (un trabajo por sección); la aceleración máxima es min(n_gpu, n_secciones).

**¿Por qué dos entornos y un archivo en disco en medio?**
CUDA 12.4 (Sonata) y CUDA 11.x (MinkowskiEngine) no conviven en un proceso. Además
Sonata está congelado: calcularlo una vez y guardarlo evita repetirlo en cada época.

**¿Por qué 5 cm de vóxel si Sonata se entrenó con 2 cm?**
5 cm reduce la nube 13 veces (8,18 M → 0,61 M en PL7) y la hace manejable. La
consecuencia es que Sonata recibe una nube más gruesa que la de su entrenamiento y
que su rejilla de 2 cm casi no elimina puntos (99,2 %). Está previsto probar 2, 3 y
5 cm y elegir con datos.

**¿Es reproducible?**
El Módulo 1 sí (sin azar; se verifica con `n_crudo`, `n_sor`, `n_voxel`). El Módulo 3
no exactamente: GridSample elige un punto al azar por celda. Por eso el protocolo
repite P2 y P3 tres veces y reporta media y desviación.

**¿Cómo saben que los descriptores sirven?**
Todavía no se sabe con una métrica de acierto (no hay etiquetas). Lo que se midió es
que no colapsan (rango efectivo 110,6 de 1088) y que la PCA y el k-means separan
estructuras de forma coherente a la vista. La prueba real será el *linear probing*.

**¿De dónde sale el 1088?**
Del up-cast: 512 (etapa 5) + 384 (etapa 4) + 192 (etapa 3).

**¿Cuántos parámetros y cómo lo verificaron?**
108,46 M: lo imprime la carga del modelo (log de Khipu) y el recuento capa por capa
con las anchuras (48, 96, 192, 384, 512) y bloques (3, 3, 3, 12, 3) da exactamente
108 461 328. El 94,3 % está en las etapas 4 y 5.

---

## 7. Código completo

### 7.1 `preprocesar.py` (Módulo 1)
```python
"""Modulo 1 del pipeline: preprocesamiento de un levantamiento LAS.

Lee una o varias nubes .las, las une en una sola seccion S_j y entrega la
nube limpia y voxelizada que consume extraer_features.py (Modulo 3).

Pasos (capitulo 2, Modulo 1):
    1. Lectura por bloques con laspy: XYZ + RGB. Las coordenadas se centran
       restando el centro de la seccion (vienen en UTM, ~10^6 m).
    2. Limpieza de ruido por SOR con q vecinos y umbral lambda (Open3D).
    3. Voxelizacion: promedio de XYZ y RGB por voxel (Open3D).
    4. Normales por PCA de los mismos q vecinos (Open3D).

Entrada : uno o varios .las (mismo sistema de coordenadas)
Salida  : <salida>.npy  float32 (N, 9) -> X Y Z R G B Nx Ny Nz
          <salida>.json metadatos (offset para volver a UTM, conteos)

Uso:
    python preprocesar.py --input datos_crudos/.../PL4_SCN0001.las \
                          --output data/PL4
"""

import argparse
import json
import time
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, nargs="+",
                    help="uno o varios .las que forman la seccion")
    ap.add_argument("--output", required=True,
                    help="ruta destino sin extension (se crean .npy y .json)")
    ap.add_argument("--voxel", type=float, default=0.05,
                    help="tamano de voxel en metros (default: 0.05)")
    ap.add_argument("--q", type=int, default=20,
                    help="vecinos para SOR y normales (default: 20)")
    ap.add_argument("--lam", type=float, default=2.0,
                    help="umbral lambda del SOR (default: 2.0)")
    ap.add_argument("--chunk", type=int, default=5_000_000,
                    help="puntos por bloque al leer el LAS")
    ap.add_argument("--prevoxel", type=float, default=0.0,
                    help="voxel previo al SOR en metros para nubes muy densas "
                         "(0 = desactivado, sigue el orden de la tesis)")
    return ap.parse_args()


def leer_las(rutas, chunk):
    """Lee los .las por bloques y devuelve XYZ (float64) y RGB (0-255)."""
    xyz, rgb = [], []
    for ruta in rutas:
        with laspy.open(ruta) as f:
            for bloque in f.chunk_iterator(chunk):
                xyz.append(np.stack([bloque.x, bloque.y, bloque.z], axis=1))
                rgb.append(np.stack([bloque.red, bloque.green, bloque.blue],
                                    axis=1).astype(np.float32))
    xyz = np.concatenate(xyz)
    rgb = np.concatenate(rgb)
    # LAS guarda el color en 16 bits; Sonata lo espera en 0-255.
    if rgb.max() > 255:
        rgb /= 256.0
    return xyz, rgb


def a_open3d(xyz, rgb):
    nube = o3d.geometry.PointCloud()
    nube.points = o3d.utility.Vector3dVector(xyz)
    nube.colors = o3d.utility.Vector3dVector(rgb / 255.0)  # Open3D usa 0-1
    return nube


def main():
    args = parse_args()
    t0 = time.time()
    meta = {"input": [str(p) for p in args.input], "voxel": args.voxel,
            "q": args.q, "lambda": args.lam, "prevoxel": args.prevoxel}

    xyz, rgb = leer_las(args.input, args.chunk)
    offset = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    xyz -= offset
    meta["offset_utm"] = offset.tolist()
    meta["n_crudo"] = int(len(xyz))
    print(f"[lectura] {len(xyz):,} puntos  ({time.time() - t0:.0f}s)", flush=True)
    nube = a_open3d(xyz, rgb)
    del xyz, rgb

    if args.prevoxel > 0:
        nube = nube.voxel_down_sample(args.prevoxel)
        meta["n_prevoxel"] = len(nube.points)
        print(f"[prevoxel] {len(nube.points):,} puntos  ({time.time() - t0:.0f}s)", flush=True)

    # SOR: descarta puntos cuya distancia media a sus q vecinos supera
    # mu + lambda * sigma.
    n_antes = len(nube.points)
    nube, _ = nube.remove_statistical_outlier(nb_neighbors=args.q,
                                              std_ratio=args.lam)
    meta["n_sor"] = len(nube.points)
    print(f"[sor] quedan {len(nube.points):,} ({n_antes - len(nube.points):,} "
          f"descartados)  ({time.time() - t0:.0f}s)", flush=True)

    # Voxelizacion: un representante por voxel (promedio de posicion y color).
    nube = nube.voxel_down_sample(args.voxel)
    meta["n_voxel"] = len(nube.points)
    print(f"[voxel] {len(nube.points):,} puntos  ({time.time() - t0:.0f}s)", flush=True)

    # Normales: autovector de menor autovalor de la covarianza de q vecinos,
    # orientadas hacia arriba para que sean consistentes.
    nube.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=args.q))
    nube.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])
    print(f"[normales] listo  ({time.time() - t0:.0f}s)", flush=True)

    salida = Path(args.output)
    salida.parent.mkdir(parents=True, exist_ok=True)
    datos = np.hstack([np.asarray(nube.points),
                       np.asarray(nube.colors) * 255.0,
                       np.asarray(nube.normals)]).astype(np.float32)
    np.save(salida.with_suffix(".npy"), datos)
    meta["segundos"] = round(time.time() - t0, 1)
    salida.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"[ok] {salida.with_suffix('.npy')}  {datos.shape}", flush=True)


if __name__ == "__main__":
    main()
```

### 7.2 `preprocesar.sbatch` (lanza el Módulo 1 en Khipu)
```bash
#!/bin/bash
# Modulo 1 del pipeline: preprocesar los .las crudos (SOR + voxel + normales).
# Solo usa CPU. Procesa cada .las de datos_crudos/ como una seccion y deja
# data/<nombre>.npy listo para extraer_features.sbatch.
#
# Uso:  sbatch preprocesar.sbatch                # todos los .las
#       sbatch preprocesar.sbatch "PL4*"         # solo los que calcen
#       VOXEL=0.03 LAM=2.5 sbatch preprocesar.sbatch   # cambiar parametros

#SBATCH --job-name=preproc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -e

REPO=/home/renzo.felix/PFC1-HyperNCD-Sonata
PATRON=${1:-*}
VOXEL=${VOXEL:-0.05}
Q=${Q:-20}
LAM=${LAM:-2.0}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sonata-env

cd "$REPO"
mkdir -p data logs

find datos_crudos -name "${PATRON}.las" | sort | while read -r las; do
    nombre=$(basename "$las" .las | tr ' ' '_' | tr -d "'")
    if [ -f "data/${nombre}.npy" ] && [ -z "$REHACER" ]; then
        echo "[skip] data/${nombre}.npy ya existe"
        continue
    fi
    echo "=== $las -> data/${nombre}.npy"
    python preprocesar.py --input "$las" --output "data/${nombre}" \
        --voxel "$VOXEL" --q "$Q" --lam "$LAM"
done
```

### 7.3 `extraer_features.py` (Módulo 3)
```python
"""Extrae features SSL de Sonata (encoder congelado) y las cachea a disco.

Es el Paso 1 del pipeline del OE2: Sonata corre una sola vez por seccion,
guarda f_ssl en disco, y HyperNCD lo consume despues desde su propio entorno
conda (las dos pilas de CUDA son incompatibles, por eso el disco de por medio).

Entrada : .txt ASCII de 9 columnas -> X Y Z R G B Nx Ny Nz
Salida  : .npz con feat (float16), coord, inverse y metadatos

Uso:
    python extraer_features.py --input data/Otros.txt \
                               --output cache/Otros_feats.npz
"""

import argparse
import time

import numpy as np
import torch

import sonata


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="nube de puntos .txt (9 columnas)")
    ap.add_argument("--output", required=True, help="destino .npz")
    ap.add_argument("--grid", type=float, default=0.02,
                    help="tamano de rejilla en metros (default: 0.02)")
    ap.add_argument("--fp32", action="store_true",
                    help="guardar en float32 en vez de float16")
    return ap.parse_args()


def cargar_nube(ruta):
    """Lee el .txt (o el .npy de preprocesar.py) y lo arma en el dict que
    espera Sonata."""
    if str(ruta).endswith(".npy"):
        datos = np.load(ruta).astype(np.float32)
    else:
        datos = np.loadtxt(ruta, dtype=np.float32)
    if datos.ndim != 2 or datos.shape[1] != 9:
        raise ValueError(
            f"{ruta}: se esperaban 9 columnas (XYZ RGB NxNyNz), "
            f"se encontraron {datos.shape}"
        )
    return {
        "coord": np.ascontiguousarray(datos[:, 0:3]),
        "color": np.ascontiguousarray(datos[:, 3:6]),
        "normal": np.ascontiguousarray(datos[:, 6:9]),
    }


def cargar_modelo():
    """Sonata preentrenado, sin flash-attn (no esta instalado en Khipu)."""
    config = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    modelo = sonata.load("sonata", repo_id="facebook/sonata",
                         custom_config=config).cuda()
    modelo.eval()
    return modelo


def upcast(point):
    """Devuelve las features a la escala de la rejilla.

    Sonata es un encoder jerarquico: la salida esta submuestreada varias veces.
    Este bloque deshace ese pooling. Los dos primeros niveles se concatenan
    (conservan el detalle fino) y el resto se propaga por indice. Es el
    procedimiento documentado en el README de Sonata.
    """
    for _ in range(2):
        assert "pooling_parent" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent
    return point


def main():
    args = parse_args()

    print(f"[1/4] leyendo {args.input}")
    t0 = time.time()
    point = cargar_nube(args.input)
    t_lectura = time.time() - t0
    n_original = point["coord"].shape[0]
    print(f"      {n_original} puntos ({t_lectura:.1f}s)")

    print("[2/4] cargando Sonata preentrenado")
    modelo = cargar_modelo()

    print(f"[3/4] transform (GridSample = {args.grid} m)")
    transform = sonata.transform.default()
    # default() trae GridSample fijo en 0.02 m; se aplica el --grid pedido.
    for t in transform.transforms:
        if type(t).__name__ == "GridSample":
            t.grid_size = args.grid
    point = transform(point)
    n_rejilla = point["coord"].shape[0]
    print(f"      {n_original} -> {n_rejilla} puntos "
          f"({100 * n_rejilla / n_original:.1f}% retenido)")

    print("[4/4] forward")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        for clave in list(point.keys()):
            if isinstance(point[clave], torch.Tensor):
                point[clave] = point[clave].cuda(non_blocking=True)
        point = modelo(point)
        point = upcast(point)

    dtype = np.float32 if args.fp32 else np.float16
    feat = point.feat.cpu().numpy().astype(dtype)
    coord = point.coord.cpu().numpy().astype(np.float32)
    # inverse mapea de la rejilla a los puntos originales: feat[inverse]
    inverse = point.inverse.cpu().numpy().astype(np.int32)

    t_forward = time.time() - t0
    vram_pico = torch.cuda.max_memory_allocated() / 1024 ** 3

    print(f"      forward en {t_forward:.1f}s")
    print(f"\n>>> FEATURES SSL: {feat.shape}   dim f_ssl = {feat.shape[1]}")
    # Cifras para la tabla de hardware de la tesis (requisitos medidos, no estimados)
    print(f">>> VRAM PICO:    {vram_pico:.2f} GB")
    print(f">>> TIEMPO:       lectura {t_lectura:.1f}s | forward {t_forward:.1f}s")
    print(f">>> RENDIMIENTO:  {n_rejilla / t_forward:,.0f} puntos/s")

    np.savez_compressed(
        args.output,
        feat=feat,
        coord=coord,
        inverse=inverse,
        grid_size=np.float32(args.grid),
        n_original=np.int64(n_original),
    )
    print(f"guardado -> {args.output}")


if __name__ == "__main__":
    main()
```

### 7.4 `extraer_features.sbatch` (lanza Sonata en Khipu)
```bash
#!/bin/bash
# Paso 1 del pipeline: cachear features SSL de Sonata.
# Uso:  sbatch extraer_features.sbatch data/Otros.txt cache/Otros_feats.npz

#SBATCH --job-name=feats
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -e

REPO=/home/renzo.felix/PFC1-HyperNCD-Sonata
ENTRADA=${1:-data/Otros.txt}
SALIDA=${2:-cache/Otros_feats.npz}

module purge
module load miniconda/3.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sonata-env

cd "$REPO"
mkdir -p cache

# Los nodos de computo no tienen salida a internet: los pesos ya estan
# cacheados en ~/.cache (descargados desde el nodo de login).
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO/sonata"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python extraer_features.py --input "$ENTRADA" --output "$SALIDA"
```

### 7.5 Fragmentos clave de Sonata (no los escribimos; se usan tal cual)
Preparación (`sonata/sonata/transform.py`, función `default`):
```python
def default():
    config = [
        dict(type="CenterShift", apply_z=True),
        dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train",
             return_grid_coord=True, return_inverse=True),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "color", "inverse"),
             feat_keys=("coord", "color", "normal")),
    ]
    return Compose(config)
```
Orden del modelo (`sonata/sonata/model.py`, `PointTransformerV3.forward`):
```python
point = Point(data_dict)
point = self.embedding(point)          # lineal 9 -> 48
point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
point.sparsify()
point = self.enc(point)                # 5 etapas con pooling entre ellas
```
Pooling (`GridPooling.forward`, resumido):
```python
grid_coord = torch.div(grid_coord, self.stride, rounding_mode="trunc")   # celda x2
feat  = segment_csr(self.proj(point.feat)[indices], idx_ptr, reduce="max")
coord = segment_csr(point.coord[indices], idx_ptr, reduce="mean")
point_dict["pooling_inverse"] = cluster    # para el up-cast
point_dict["pooling_parent"]  = point
```
