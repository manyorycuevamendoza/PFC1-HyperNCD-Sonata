# M2: ¿colapsa por dominio? · semposs_e8_vs_arch

Checkpoint: `None` · voxel 0.05 m

| Escena | Puntos | Cabeza dominante (%) | Cabezas > 1 % | Rango efectivo f_sem | NMI | ARI | mIoU 1 a 1 |
|---|---|---|---|---|---|---|---|
| capilla_SemanticPOSS_e8 | 1,135,946 | 98.4 | 1 | 3.4 | 0.018 | 0.003 | 0.114 |
| capilla_SemanticPOSS_e8 (sin «otro») | | | | | 0.027 | 0.004 | 0.077 |
| capilla_ArCH_best_e21 | 1,135,946 | 44.5 | 9 | 1.7 | 0.192 | 0.022 | 0.179 |
| capilla_ArCH_best_e21 (sin «otro») | | | | | 0.437 | 0.447 | 0.336 |
| capilla_ArCH_last_e30 | 1,135,946 | 44.6 | 9 | 1.7 | 0.191 | 0.030 | 0.183 |
| capilla_ArCH_last_e30 (sin «otro») | | | | | 0.439 | 0.461 | 0.355 |
