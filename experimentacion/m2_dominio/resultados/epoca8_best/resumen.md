# M2: ¿colapsa por dominio? · epoca8_best

Checkpoint: `/content/drive/MyDrive/PFC1_HyperNCD_Sonata/m2/entrada/datos_m2_descomprimidos/hyperncd/checkpoints/SemanticPOSS/split0_run4/SemanticPOSS_S0_best-epoch=8-step=11196.ckpt` · voxel 0.05 m

| Escena | Puntos | Cabeza dominante (%) | Cabezas > 1 % | Rango efectivo f_sem | NMI | ARI | mIoU 1 a 1 |
|---|---|---|---|---|---|---|---|
| semposs_val_03 | - | 30.4 | 7 | 5.4 | 0.478 | 0.413 | 0.375 |
| semposs_train_00 | - | 28.2 | 6 | 4.1 | 0.493 | 0.362 | 0.472 |
| capilla | 1,135,946 | 98.4 | 1 | 3.4 | 0.018 | 0.003 | 0.114 |
| capilla (sin «otro») | | | | | 0.027 | 0.004 | 0.077 |
| capilla_80k | 80,000 | 91.8 | 4 | 1.6 | 0.033 | -0.064 | 0.099 |
| capilla_80k (sin «otro») | | | | | 0.019 | -0.010 | 0.068 |
| PL13 | 803,099 | 82.5 | 4 | 2.1 | - | - | - |
| PL7 | 610,637 | 81.4 | 4 | 2.1 | - | - | - |
