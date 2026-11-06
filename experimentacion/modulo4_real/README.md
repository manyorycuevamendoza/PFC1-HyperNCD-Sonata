# Módulo 4 real (PL13)

Une `f_sem` (M2, `cache/PL13_fsem.npz`) y `f_ssl` (M3, `datos/modulo3_PL13/PL13_feats.npz`)
por punto original y corre las fórmulas del Módulo 4 de `modulo4_simulado/`.
Regiones = celdas de 0,75 m (DBSCAN daría una sola región en una nave conectada).

```
python experimentacion/modulo4_real/correr_modulo4_real.py \
    --npy experimentacion/datos/modulo3_PL13/PL13_SCN0001.npy \
    --fsem cache/PL13_fsem.npz \
    --fssl experimentacion/datos/modulo3_PL13/PL13_feats.npz \
    --salida experimentacion/resultados/modulo4_real
python experimentacion/modulo4_real/graficar_modulo4_real.py
```

Sin etiquetas de la Catedral no hay mIoU: se compara el acuerdo (ARI/NMI) entre
M2, M3, M4 base y M4, la estabilidad entre semillas, la silueta y el perfil de
cada grupo (altura, normal vertical). Salidas en `experimentacion/resultados/modulo4_real/`.

**Limitación:** el `f_sem` salió del checkpoint `split0_full ... epoch=0` (casi sin
entrenar), no del mejor (`split0_run4 ... epoch=8`, mIoU 0.38). Repetir M2 con ese
checkpoint en Colab antes de sacar conclusiones sobre el aporte de M2.
