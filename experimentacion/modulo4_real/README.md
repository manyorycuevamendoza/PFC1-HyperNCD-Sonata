# Módulo 4 real (PL13)

Une `f_sem` (M2, `cache/PL13_fsem.npz`) y `f_ssl` (M3, `datos/modulo3_PL13/PL13_feats.npz`)
por punto original y corre las fórmulas del Módulo 4 de `modulo4_simulado/`
(afinidad S_b, hiperaristas, prototipos y Sinkhorn semi-relajado).

Regiones = celdas de 0,75 m (DBSCAN daría una sola región en una nave conectada).
Solo PL13 tiene M2 y M3 a la vez; PL7 no tiene `f_ssl` aquí.

## Observaciones del autor (cualitativas, a partir del visor 3D)

> Captura del visor de M4 en PL13 (K = 8). La variante exacta (original o adaptado) **no quedó
> registrada**; confirmar antes de citarla. No hay medida numérica de esto todavía.

1. **Lo que funciona.** El descriptor distingue morfologías con rigor. Los anillos de la cúpula salen
   como grupos separados y coherentes (naranja y marrón). Aunque sean grupos distintos, siguen siendo
   parte de la misma cúpula: es una separación aceptable y **no se quiere perder esa propiedad**.
2. **Lo que falla: sobre-segmentación de superficies curvas.** Los techos curvos (bóvedas) se parten
   en varios grupos, con muchos parches pequeños (azules y de otros colores) mezclados. Son una sola
   estructura, pero como su orientación cambia, el método los trata como geometrías muy lejanas y las
   asigna a clases distintas.
3. **Hipótesis (sin verificar).** (a) las regiones son celdas fijas de 0,75 m que cortan la superficie
   de forma arbitraria; (b) el descriptor depende de la orientación y la curvatura local (normales),
   que cambian a lo largo de una bóveda; (c) nada obliga a que regiones vecinas y parecidas compartan
   grupo (no hay restricción de continuidad espacial).

**Para la tesis:** incluir esto como limitación observada de M4 y como motivación de la mejora de
continuidad espacial. Ideas para probarlo, en `ESTADO_DEL_PROYECTO.md` ("Ideas para mejorar M4").

## Cómo correrlo

```
python experimentacion/modulo4_real/correr_modulo4_real.py \
    --npy experimentacion/datos/modulo3_PL13/PL13_SCN0001.npy \
    --fsem cache/PL13_fsem.npz \
    --fssl experimentacion/datos/modulo3_PL13/PL13_feats.npz \
    --salida experimentacion/resultados/modulo4_real
python experimentacion/modulo4_real/graficar_modulo4_real.py
python experimentacion/modulo4_real/visor_modulo4_real.py
```

Tarda ~80 s en la laptop, sin GPU. Para `PL13_fsem.npz` ver "M2 en Colab" en
`ESTADO_DEL_PROYECTO.md`: debe venir del checkpoint `split0_run4 ... epoch=8`.

## Variantes que compara

| Variante | Descriptor |
|---|---|
| M2 | solo `f_sem` |
| M3 | solo `f_ssl` |
| M4 base | `[f_sem ; f_geo]` (línea base de HyperNCD) |
| M4 | `[f_sem ; f_ssl]` (propuesta) |

## Qué mide (y qué no)

La Catedral **no tiene etiquetas**, así que no hay mIoU. Se mide acuerdo entre variantes
(ARI/NMI), estabilidad entre semillas, silueta y el perfil de cada grupo (altura, normal
vertical). Eso dice si las variantes se parecen o son estables, **no cuál acierta más**.
Para eso hay un recorte en etiquetado: `experimentacion/etiquetado/`.

## Resultados (K = 4, `f_sem` de la época 8)

| | Estabilidad (ARI) | Acuerdo con M4 (ARI) |
|---|---|---|
| M2 | 0.53 | 0.17 |
| M3 | 0.63 | 0.46 |
| M4 base | 0.64 | 0.14 |
| M4 | 0.55 | — |

Con el `f_sem` de la época 0 (`../resultados/modulo4_real_m2_epoch0/`), M4 era casi
una copia de M3 (ARI 0.87, estabilidad 0.95). Con la época 8 mezcla las dos fuentes pero
es menos estable.

## Pendiente

- Calcular mIoU y confusión cuando esté el recorte etiquetado.
- Probar otras fusiones de M2 y M3 (pesos, PCA de `f_ssl`, fusión tardía de afinidades,
  confianza de M2 por región, dos hipergrafos).
