# Prompt para ejecutar las pruebas de parámetros en Khipu

Copia todo lo que sigue y pégalo como primer mensaje en el Claude del compañero
que tiene acceso a Khipu, con el repositorio clonado.

---

## Contexto

Repositorio: `PFC1-HyperNCD-Sonata`. Tesis sobre descubrimiento de clases
novedosas en nubes de puntos de la Catedral de Lima. El flujo tiene dos
programas:

**`preprocesar.py`** (Módulo 1, CPU) convierte un LAS en una matriz `S` de N×9
(posición, color, normal) en seis pasos: carga por bloques, centrado, filtrado
SOR, voxelización, normales por PCA, escritura. Parámetros: `--chunk` (5e6),
`--q` (20, se usa a la vez para el SOR **y** para las normales), `--lam` (2.0),
`--voxel` (0.05).

**`extraer_features.py`** (Módulo 3, GPU) pasa `S` por el codificador PTv3 de
Sonata congelado y produce `f_ssl` de 1088 dimensiones. Parámetros: `--grid`
(0.02), `--fp32` (si no, guarda en float16). El tamaño de parche está fijo en
1024 en `cargar_modelo()`, y FlashAttention está desactivada porque no está
instalada en el clúster.

**El problema:** ninguno de esos valores se eligió midiendo. Son los defaults
del programa o del método original. El asesor pide justificar cada uno: por qué
ese valor y no más o menos, y qué pasa si el paso no se ejecuta.

Escaneo de prueba: `PL13_SCN0001.las` (en Khipu se llama `PL13'_SCN0001.las`,
con apóstrofo). 8 239 904 puntos. Con los defaults: 8 020 796 tras el SOR
(2,66 % descartado) y 803 099 representantes tras voxelizar.

## Primero: el script de métricas (no existe, hay que escribirlo)

Crea `experimentacion/pruebas_parametros/medir_calidad.py`. Recibe un `.npy` del
Módulo 1 y calcula:

1. **Rugosidad residual `r`**: por cada representante, ajusta un plano a sus 20
   vecinos (PCA: el plano es el de los dos autovectores mayores) y toma la
   distancia del punto a ese plano. Media en **milímetros**.
2. **Dispersión angular**: ángulo entre la normal de cada punto y la normal
   media de sus 20 vecinos. Las normales están orientadas a `n_z ≥ 0`, así que
   usa `arccos(|cos|)`. Media en **grados**.
3. **Las dos anteriores separadas por densidad**: clasifica los representantes
   por distancia media a sus 20 vecinos y reporta aparte el decil más disperso
   ("zonas ralas"). La hipótesis es que las diferencias se concentran ahí.
4. **Las dos anteriores separadas por planitud** (solo para P3): usa la razón
   de autovalores λ₃/(λ₁+λ₂+λ₃) para separar zonas planas de aristas.

Usa muestra aleatoria con semilla fija (200 000 puntos) si el cálculo completo
es lento, pero **la misma muestra espacial en todas las condiciones** (empareja
por vecino más cercano contra la condición de referencia).

Y `experimentacion/pruebas_parametros/comparar_features.py`, que recibe dos
`.npz` y devuelve **ARI y NMI**: k-means con k=8 sobre `f_ssl` normalizado
(mismo procedimiento que `experimentacion/modulo3_sonata/visualizar_features.py`:
restar la media de una muestra de 100 000 puntos con semilla 0, normalizar a
largo 1, 30 iteraciones) en cada uno, y comparar los dos agrupamientos.

**Problema a resolver:** cada condición tiene distinto número de puntos, así que
los agrupamientos no son comparables punto a punto. Empareja por vecino más
cercano contra la nube de referencia (KD-tree, descartando parejas a más de
5 cm) y calcula ARI/NMI sobre los emparejados. Reporta cuántos quedaron.

## Las seis pruebas

Ejecútalas en este orden. P0 primero: si falla, las demás no son fiables.

### P0 — Invariancia al tamaño de bloque (CPU, 3 corridas)

`--chunk` en {1e6, 5e6, 20e6}, resto por defecto. **Resultado esperado: ninguna
diferencia.** Verifica que `n_crudo`, `n_sor` y `n_voxel` del `.json` coincidan
exactamente y que las matrices `S` sean iguales punto a punto. Si no coinciden,
para y repórtalo: hay una dependencia del bloque que corregir antes de seguir.

### P1 — Necesidad del filtrado SOR y sus parámetros (CPU, 8 corridas)

| Condición | Orden |
|---|---|
| Sin filtrado | `--q 20 --lam 1e9` — **verifica que `n_sor == n_crudo`**; si no, añade una bandera `--sin-sor` al programa |
| q=10 | `--q 10 --lam 2` |
| q=20 (referencia) | `--q 20 --lam 2` |
| q=30 | `--q 30 --lam 2` |
| q=40 | `--q 40 --lam 2` |
| λ=1.5 | `--q 20 --lam 1.5` |
| λ=2.5 | `--q 20 --lam 2.5` |
| λ=3 | `--q 20 --lam 3` |

Todas con `--voxel 0.05`.

**Ojo:** `--q` cambia a la vez el SOR y las normales, así que en las filas de
q≠20 el efecto medido es conjunto. Esto se resuelve en P3.

### P2 — Tamaño del vóxel y rejilla de Sonata (CPU + GPU)

`--voxel` en {0.02, 0.03, 0.05, 0.10} con `--grid 0.02` fijo. Para el mejor
`--voxel`, repite con `--grid 0.04`.

Además de las métricas comunes, reporta el número de representantes (determina
si la sección cabe en la GPU) y el porcentaje retenido por la rejilla de Sonata
(indica cuál de las dos resoluciones manda).

**Con `--voxel 0.02` la nube será ~4× más grande que la de referencia y puede no
caber en los 48 GB de la A6000.** Si falla por memoria, repórtalo como resultado
en lugar de reducir la nube.

### P3 — Vecinos de las normales (requiere cambio de código)

Separa los dos usos de `--q`: añade `--q-normales` a `preprocesar.py`, con
default igual a `--q` para no romper nada. Pásalo a
`estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=args.q_normales))`.

Luego barre `--q-normales` en {10, 20, 30, 40} con `--q 20 --lam 2` fijo. El
indicador que manda es la dispersión angular, **reportada por separado en zonas
planas y en aristas** (los dos extremos fallan de forma opuesta y un promedio
global los cancelaría).

### P4 — Precisión de almacenamiento (GPU, 2 corridas)

`extraer_features.py` con y sin `--fp32` sobre la misma `S`. Compara con ARI,
NMI y tamaño del archivo.

### P5 — Tamaño de parche del codificador (GPU, 3 corridas)

El parche está fijo en `cargar_modelo()`:
`config = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)`.
Hazlo parámetro (`--parche`) y barre {512, 1024, 2048}. Mide **VRAM pico**,
tiempo, ARI y NMI.

Contexto: con 1024 y 796 315 puntos el pico fue 34,60 GB de 48 GB, muy por
encima de la cota analítica de 4,2 GB. Si 512 baja la memoria sin cambiar el
agrupamiento, es una alternativa a dividir las secciones grandes en bloques.
Con 2048 probablemente no quepa; si falla, repórtalo como resultado.

## Entregable

Un `.csv` por prueba en `experimentacion/pruebas_parametros/resultados/`, con
una fila por condición:

```
prueba, condicion, q, q_normales, lam, voxel, grid, parche, precision,
n_sor, n_voxel, pct_descartado, segundos, vram_gb,
r_mm_global, r_mm_ralas, disp_grados_global, disp_grados_ralas,
disp_grados_planas, disp_grados_aristas,
ari_vs_ref, nmi_vs_ref, puntos_emparejados
```

Más los logs de cada corrida. Deja en blanco lo que no aplique a esa prueba.

## Lo que NO hay que hacer

- **No hay etiquetas de verdad de terreno** para esta sección. No intentes medir
  mIoU ni accuracy. El *linear probing* queda fuera.
- **No estimes ningún valor que no hayas medido.** Si una corrida falla o una
  métrica no se puede calcular, dilo en el entregable.
- **No cambies dos parámetros a la vez**, salvo donde el prompt lo indica
  explícitamente (P1 con q≠20, que ya está advertido).
- No toques los defaults del programa. Los cambios de código de P3 y P5 son
  parámetros nuevos con default igual al comportamiento actual.

---

## Lo que hace falta para acceder a Khipu

- Cuenta en el clúster Khipu de UTEC y clave SSH registrada.
- Repositorio clonado y el entorno `sonata-env` creado con el guion de
  instalación. Estas pruebas solo tocan el Módulo 1 (`preprocesar.py`, CPU) y el
  Módulo 3 (`extraer_features.py`, GPU); no hace falta `hyperncd-env`.
- Los archivos LAS en `datos_crudos/` (no están en el repositorio).
- La variable `REPO` al inicio de cada `.sbatch` apuntando a la copia del
  repositorio del usuario.
- Pesos de Sonata descargados desde el nodo de acceso (los nodos con GPU no
  tienen salida a internet).

## Costo estimado

- P0, P1, P3: CPU, ~1 min por corrida. 15 corridas.
- P2, P4, P5: GPU A6000, ~1 min por corrida más la cola. ~12 corridas.
- El script de métricas es lo más lento: la búsqueda de 20 vecinos sobre
  800 000 puntos, por cada condición.
