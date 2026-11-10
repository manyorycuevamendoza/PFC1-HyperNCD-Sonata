# M2 de SemanticPOSS contra M2 de ArCH, capilla etiquetada (sin `otro`)

mIoU en %, asignación 1 a 1 (y por mayoría entre paréntesis), K = 8. Misma escena, mismo evaluador.

| Método | SemanticPOSS + Sonata original | **ArCH** + Sonata original | SemanticPOSS + Sonata adaptado | **ArCH** + Sonata adaptado |
|---|---|---|---|---|
| M2 | 34.1 (34.2) | 49.0 (49.1) | 34.1 (34.2) | 49.0 (49.1) |
| M3 | 53.4 (48.2) | 52.5 (47.3) | 35.6 (37.3) | 35.1 (35.0) |
| M4_base | 32.1 (29.9) | 52.9 (50.6) | 32.1 (29.9) | 52.9 (50.6) |
| M4 | 52.5 (48.9) | 50.6 (50.8) | 36.1 (33.4) | 47.4 (40.6) |

## IoU por clase (%), 1 a 1, K = 8, sin `otro`

| Clase | M3 · Sema · orig | M3 · ArCH · orig | M4_base · Sema · orig | M4_base · ArCH · orig | M4 · Sema · orig | M4 · ArCH · orig | M4 · Sema · adap | M4 · ArCH · adap |
|---|---|---|---|---|---|---|---|---|
| arco | 18.4 | 18.7 | 13.9 | 49.8 | 18.8 | 42.3 | 19.6 | 27.2 |
| columna | 32.1 | 28.9 | 8.7 | 22.1 | 28.2 | 28.6 | 23.8 | 24.3 |
| cupula | 81.0 | 80.5 | 30.0 | 57.1 | 80.5 | 43.0 | 41.0 | 43.9 |
| pared | 69.1 | 68.2 | 56.1 | 69.0 | 69.6 | 74.2 | 52.8 | 72.0 |
| techo | 66.3 | 66.4 | 51.7 | 66.6 | 65.3 | 65.1 | 43.2 | 69.8 |

Columnas: método · M2 (Semp = SemanticPOSS) · Sonata (orig = original, adap = adaptado).
