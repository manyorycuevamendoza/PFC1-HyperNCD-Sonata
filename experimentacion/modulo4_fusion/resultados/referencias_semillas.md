# M2, M3, M4 base y M4 con 10 semillas (K = 8, mIoU 1 a 1, sin `otro`)

Media ± desviación sobre semillas de inicialización (y [mín, máx]). Misma escena, mismas etiquetas.

| Combinación | M2 | M3 | M4 base (`f_geo`) | M4 (`f_sem` + `f_ssl`) |
|---|---|---|---|---|
| HyperNCD original + Sonata original | 35.9 ± 0.6 [35, 37] | 45.4 ± 8.5 [33, 53] | 32.5 ± 0.6 [32, 34] | 43.4 ± 8.2 [32, 52] |
| HyperNCD original + Sonata adaptado | 35.9 ± 0.6 [35, 37] | 34.3 ± 1.7 [32, 38] | 32.5 ± 0.6 [32, 34] | 33.7 ± 2.1 [29, 36] |
| HyperNCD adaptado + Sonata original | 48.4 ± 1.1 [46, 50] | 45.4 ± 8.5 [33, 53] | 51.4 ± 1.7 [48, 54] | 52.5 ± 2.5 [50, 56] |
| HyperNCD adaptado + Sonata adaptado | 48.4 ± 1.1 [46, 50] | 34.3 ± 1.7 [32, 38] | 51.4 ± 1.7 [48, 54] | 47.0 ± 0.8 [46, 48] |
